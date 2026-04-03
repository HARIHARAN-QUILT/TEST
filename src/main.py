"""
AI-Powered Document Analysis & Extraction API
Supports PDF, DOCX, and Image formats with Tesseract OCR and OpenRouter API
"""

import os
import base64
import json
import re
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pytesseract
from pdf2image import convert_from_bytes
from docx import Document
from PIL import Image
import requests

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Document Analysis API",
    description="Extract, analyze, and summarize content from various document formats",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
API_KEY = os.getenv("API_KEY", "sk_track2_default")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")


class DocumentRequest(BaseModel):
    fileName: str
    fileType: str
    fileBase64: str


class DocumentResponse(BaseModel):
    status: str
    fileName: str
    summary: str
    entities: dict
    sentiment: str


def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        images = convert_from_bytes(file_bytes)
        extracted_text = ""
        for image in images:
            text = pytesseract.image_to_string(image)
            extracted_text += text + "\n"
        return extracted_text.strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        doc = Document(BytesIO(file_bytes))
        extracted_text = "\n".join([p.text for p in doc.paragraphs])
        return extracted_text.strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text from DOCX: {str(e)}")


def extract_text_from_image(file_bytes: bytes) -> str:
    try:
        image = Image.open(BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text from image: {str(e)}")


def extract_text(file_type: str, file_bytes: bytes) -> str:
    if file_type.lower() == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif file_type.lower() == "docx":
        return extract_text_from_docx(file_bytes)
    elif file_type.lower() in ["png", "jpg", "jpeg"]:
        return extract_text_from_image(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def analyze_with_openrouter(text: str) -> dict:
    if not text or len(text.strip()) == 0:
        raise ValueError("No text extracted from document")

    prompt = f"""Analyze the following document text and provide:
1. A concise summary (2-3 sentences)
2. Named entities (names, dates, organizations, amounts)
3. Overall sentiment (Positive, Negative, or Neutral)

Document text:
{text[:3000]}

Respond in this exact JSON format:
{{
    "summary": "your summary here",
    "entities": {{
        "names": ["name1", "name2"],
        "dates": ["date1", "date2"],
        "organizations": ["org1", "org2"],
        "amounts": ["amount1", "amount2"]
    }},
    "sentiment": "Neutral"
}}

Important: Return ONLY valid JSON, no additional text."""

    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "mistralai/mixtral-8x7b-instruct",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1024
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            print(f"[OpenRouter] API Error {response.status_code}: {response.text}")

        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                analysis = json.loads(content)
        except json.JSONDecodeError:
            analysis = {
                "summary": content[:200],
                "entities": {"names": [], "dates": [], "organizations": [], "amounts": []},
                "sentiment": "Neutral"
            }

        return analysis

    except requests.exceptions.RequestException as e:
        raise ValueError(f"OpenRouter API request failed: {str(e)}")
    except (KeyError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to parse OpenRouter response: {str(e)}")


def validate_entities(entities: dict) -> dict:
    required_keys = ["names", "dates", "organizations", "amounts"]
    validated = {}

    for key in required_keys:
        if key in entities and isinstance(entities[key], list):
            validated[key] = [str(i).strip() for i in entities[key] if i]
        else:
            validated[key] = []

    return validated


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "Document Analysis API is running"
    }


@app.post("/api/document-analyze", response_model=DocumentResponse)
async def analyze_document(
    request: DocumentRequest,
    x_api_key: str = Header(None)
):

    # API key validation
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing API key"
        )

    try:

        # Validate supported fileType
        if request.fileType.lower() not in ["pdf", "docx", "png", "jpg", "jpeg"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {request.fileType}"
            )

        # NEW VALIDATION (filename extension must match fileType)
        file_extension = Path(request.fileName).suffix.replace(".", "").lower()
        file_type = request.fileType.lower()

        if file_extension != file_type:
            raise HTTPException(
                status_code=400,
                detail="Invalid filetype or extension"
            )

        # Decode Base64
        try:
            file_bytes = base64.b64decode(request.fileBase64)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid Base64 encoding: {str(e)}"
            )

        # Extract text
        extracted_text = extract_text(request.fileType, file_bytes)

        if not extracted_text:
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the document"
            )

        # AI analysis
        analysis = analyze_with_openrouter(extracted_text)

        entities = validate_entities(analysis.get("entities", {}))

        sentiment = analysis.get("sentiment", "Neutral").strip()
        if sentiment not in ["Positive", "Negative", "Neutral"]:
            sentiment = "Neutral"

        return DocumentResponse(
            status="success",
            fileName=request.fileName,
            summary=analysis.get("summary", "Unable to generate summary"),
            entities=entities,
            sentiment=sentiment
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/api/document-analyze/batch")
async def analyze_documents_batch(
    requests_list: list[DocumentRequest],
    x_api_key: str = Header(None)
):

    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing API key"
        )

    results = []

    for request in requests_list:
        try:
            result = await analyze_document(request, x_api_key)
            results.append(result)
        except HTTPException as e:
            results.append({
                "status": "error",
                "fileName": request.fileName,
                "error": e.detail
            })

    return {"documents": results}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )