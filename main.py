from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
import requests
from dotenv import load_dotenv
import os
from typing import Optional
import json
from groq import Groq  # Ensure you have the groq package installed

# Load environment variables
load_dotenv()

app = FastAPI()

# In-memory storage for various types of content
content_store = []         # raw webhook payloads
hospital_store = []        # data forwarded to hospital API
consultation_store = []    # data forwarded to consultation API

# Helper functions to simulate forwarding the data to external APIs
def forward_to_hospital(data: dict) -> dict:
    hospital_store.append(data)
    return {"message": "Data forwarded to hospital."}

def forward_to_consultation(data: dict) -> dict:
    consultation_store.append(data)
    return {"message": "Data forwarded to consultation."}


async def make_bolna_call(recipient_phone_number: str):
    """Make a call to the Bolna API"""
    url = "https://api.bolna.dev/call"
    payload = {
        "agent_id": os.getenv("agent_id"),
        "recipient_phone_number": recipient_phone_number,
    }
    headers = {
        "Authorization": f"Bearer {os.getenv('Authorization')}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"API call failed: {str(e)}")


@app.post("/make-call")
async def initiate_call(request: Request, recipient_phone_number: str = Query(...)):
    """Endpoint to initiate a call with a recipient phone number passed as a query parameter"""
    try:
        result = await make_bolna_call(recipient_phone_number)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhook")
async def webhook(request: Request):
    """
    Handle incoming webhook data.
    The data includes a conversation transcript and extracted information.
    This endpoint:
      1. Stores the full payload.
      2. Sends the transcript, extracted data, and context details to the Groq API.
      3. Parses Groq's response to decide if the situation is an emergency.
      4. Forwards the data to either a hospital or consultation API accordingly.
    """
    try:
        data = await request.json()
        print(data)

        # Store the incoming webhook data for record-keeping.
        content_store.append(data)

        # Only process if 'extracted_data' is available.
        if "extracted_data" in data:
            # Build a prompt for Groq.
            # The prompt instructs Groq to analyze the transcript along with extracted and context data,
            # and then output a JSON object with the keys:
            #   - status: "emergency" or "not emergency"
            #   - location
            #   - issue
            #   - recipient_phone_number
            groq_prompt = (
                "You are an emergency evaluation assistant. Analyze the following emergency call data "
                "and determine if it is an emergency. Return a JSON object with the following keys:\n"
                "- status (with values 'emergency' or 'not emergency')\n"
                "- location\n"
                "- issue\n"
                "- recipient_phone_number\n\n"
                f"Transcript: {data.get('transcript', '')}\n"
                f"Extracted Data: {json.dumps(data.get('extracted_data', {}))}\n"
                f"Context Details: {json.dumps(data.get('context_details', {}))}\n"
            )

            # Initialize the Groq client and send the request.
            client = Groq()
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an emergency evaluation assistant."
                    },
                    {
                        "role": "user",
                        "content": groq_prompt,
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.5,
                max_completion_tokens=1024,
                top_p=1,
                stop=None,
                stream=False,
            )

            groq_response_text = chat_completion.choices[0].message.content
            try:
                groq_response = json.loads(groq_response_text)
            except Exception as e:
                # In case parsing fails, default to non-emergency using available data.
                groq_response = {
                    "status": "not emergency",
                    "location": data.get("extracted_data", {}).get("location", ""),
                    "issue": data.get("extracted_data", {}).get("issues", ""),
                    "recipient_phone_number": data.get("context_details", {}).get("recipient_phone_number", "")
                }

            # Based on Groq's evaluation, forward the data.
            if groq_response.get("status", "").lower() == "emergency":
                forward_result = forward_to_hospital(groq_response)
            else:
                forward_result = forward_to_consultation(groq_response)

            # Attach Groq's response and the forwarding result to the original data.
            data["groq_response"] = groq_response
            data["forward_result"] = forward_result

        return {"status": "success", "message": "Webhook received and processed", "data": data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON data: {str(e)}")


@app.post("/hospital")
async def hospital_endpoint(request: Request):
    """
    Simulated Hospital API endpoint.
    Receives emergency data and stores it.
    """
    try:
        data = await request.json()
        hospital_store.append(data)
        return {"status": "success", "message": "Data received by hospital API", "data": data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON data: {str(e)}")


@app.post("/consultation")
async def consultation_endpoint(request: Request):
    """
    Simulated Consultation API endpoint.
    Receives non-emergency data and stores it.
    """
    try:
        data = await request.json()
        consultation_store.append(data)
        return {"status": "success", "message": "Data received by consultation API", "data": data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON data: {str(e)}")


@app.get("/display", response_class=HTMLResponse)
async def display_content():
    """
    Display the stored webhook, hospital, and consultation data in an HTML page.
    """
    html_content = """
    <html>
        <head>
            <title>Received Content</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .content-item { margin: 10px 0; padding: 10px; border: 1px solid #ccc; }
                .section { margin-bottom: 30px; }
            </style>
        </head>
        <body>
            <h1>Received Content</h1>
            <div class="section">
                <h2>Webhook Data</h2>
    """
    for item in content_store:
        html_content += f'<div class="content-item"><pre>{json.dumps(item, indent=2)}</pre></div>'

    html_content += """
            </div>
            <div class="section">
                <h2>Hospital Data</h2>
    """
    for item in hospital_store:
        html_content += f'<div class="content-item"><pre>{json.dumps(item, indent=2)}</pre></div>'

    html_content += """
            </div>
            <div class="section">
                <h2>Consultation Data</h2>
    """
    for item in consultation_store:
        html_content += f'<div class="content-item"><pre>{json.dumps(item, indent=2)}</pre></div>'

    html_content += """
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
