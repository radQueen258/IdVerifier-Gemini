from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
from PIL import Image
import io
import json
import re
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ID Card Verifier via Gemini AI",
    description="Verify ID cards using Google Gemini Vision API",
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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_gemini_api_key_here")  # Set in environment variables

# Initialize Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")

class VerificationResponse(BaseModel):
    is_id_card: bool
    confidence: float
    document_type: str
    side: str = "unknown"
    reason: str = ""

class ErrorResponse(BaseModel):
    error: str
    detail: str = ""

@app.get("/")
async def root():
    return {
        "message": "ID Card Verification API using Gemini AI",
        "status": "healthy",
        "endpoints": {
            "verify": "POST /verify - Verify single ID card",
            "health": "GET / - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Gemini connection by listing models
        models = genai.list_models()
        gemini_status = "connected"
    except Exception as e:
        gemini_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "gemini_api": gemini_status,
        "timestamp": "2024-01-01T00:00:00Z"  # You might want to use actual timestamp
    }

def analyze_with_gemini(image: Image.Image) -> dict:
    """
    Analyze image using Gemini Vision API
    """
    try:
        # Use Gemini Pro Vision model
        model = genai.GenerativeModel('gemini-pro-vision')
        
        prompt = """
        Analyze this image and determine if it's an identity document. 
        Consider the following document types:
        - National ID card (front/back)
        - Driver's license (front/back) 
        - Passport
        - Other identity documents
        
        Look for these features:
        - Official government seals/logos
        - Personal photo
        - Identification numbers
        - Personal information (name, date of birth, etc.)
        - Security features (holograms, microprint)
        - Standard document layout and formatting
        
        Respond with ONLY valid JSON format, no other text:
        {
            "is_id_card": true/false,
            "confidence": 0.0-1.0,
            "document_type": "national_id_front" | "national_id_back" | "driver_license_front" | "driver_license_back" | "passport" | "other",
            "side": "front" | "back" | "unknown",
            "reason": "brief explanation of your analysis"
        }
        
        Be accurate and conservative in your assessment.
        """
        
        # Generate content
        response = model.generate_content([prompt, image])
        response_text = response.text.strip()
        
        # Clean the response - remove markdown code blocks if present
        response_text = re.sub(r'```json\s*|\s*```', '', response_text)
        response_text = response_text.strip()
        
        logger.info(f"Gemini raw response: {response_text}")
        
        # Parse JSON response
        result = json.loads(response_text)
        
        # Validate required fields
        required_fields = ["is_id_card", "confidence", "document_type", "reason"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        # Fallback: try to extract information from text response
        return {
            "is_id_card": False,
            "confidence": 0.0,
            "document_type": "other",
            "side": "unknown",
            "reason": f"Failed to parse AI response: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Gemini analysis error: {e}")
        raise

@app.post("/verify", response_model=VerificationResponse)
async def verify_id_card(file: UploadFile = File(...)):
    """
    Verify if uploaded image is an ID card using Gemini AI
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image (JPEG, PNG, etc.)"
            )
        
        # Check file size (max 10MB)
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(
                status_code=400,
                detail="Image size too large. Maximum size is 10MB."
            )
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        logger.info(f"Processing image: {file.filename}, size: {image.size}")
        
        # Analyze with Gemini
        result = analyze_with_gemini(image)
        
        return VerificationResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verification error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process image: {str(e)}"
        )

@app.post("/verify-batch")
async def verify_batch(files: list[UploadFile] = File(...)):
    """
    Verify multiple ID card images at once
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files allowed per batch request"
        )
    
    results = []
    for file in files:
        try:
            # Create a simple wrapper for single file verification
            class SingleFile:
                def __init__(self, file_obj):
                    self.file = file_obj
                    self.filename = file_obj.filename
                    self.content_type = file_obj.content_type
                
                async def read(self):
                    return await self.file.read()
            
            result = await verify_id_card(SingleFile(file))
            results.append({
                "filename": file.filename,
                "status": "success",
                "result": result.dict()
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "processed_files": len(results),
        "results": results
    }

@app.post("/verify-detailed")
async def verify_detailed(file: UploadFile = File(...)):
    """
    Get detailed analysis of the ID document
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        model = genai.GenerativeModel('gemini-pro-vision')
        
        detailed_prompt = """
        Analyze this identity document in detail and provide comprehensive information.
        
        Please provide a detailed JSON response with the following structure:
        {
            "verification": {
                "is_identity_document": true/false,
                "document_type": "national_id" | "driver_license" | "passport" | "other",
                "side": "front" | "back" | "unknown",
                "confidence": 0.0-1.0,
                "country_origin": "estimated country or unknown"
            },
            "detected_features": {
                "has_photo": true/false,
                "has_signature": true/false,
                "has_barcode": true/false,
                "has_hologram": true/false,
                "has_machine_readable_zone": true/false
            },
            "content_analysis": {
                "clarity": "excellent" | "good" | "poor" | "unreadable",
                "completeness": "complete" | "partial" | "cropped",
                "detected_text_elements": ["name", "id_number", "birth_date", "etc"]
            },
            "security_assessment": {
                "appears_genuine": true/false,
                "potential_issues": ["blurry", "glare", "low_quality", "suspicious_layout"]
            }
        }
        
        Be thorough and objective in your analysis.
        """
        
        response = model.generate_content([detailed_prompt, image])
        response_text = response.text.strip()
        response_text = re.sub(r'```json\s*|\s*```', '', response_text)
        
        detailed_result = json.loads(response_text)
        
        return {
            "filename": file.filename,
            "detailed_analysis": detailed_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

@app.exception_handler(400)
async def bad_request_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Bad request", "detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
