import os
import shutil
import uuid
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.hybrid_agent import HybridAgent
from src.report_analyzer import ReportAnalyzer
from src.ml_engine import MLEngine
from src.utils import logger

# Initialize FastAPI
app = FastAPI(
    title="AI Agent for Personalized Treatment Recommendation",
    description="Backend API service providing NLP, ML, GenAI, and report OCR capabilities.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
try:
    hybrid_agent = HybridAgent()
    report_analyzer = ReportAnalyzer()
    ml_engine = MLEngine()
except Exception as e:
    logger.error(f"Failed to initialize clinical engines: {e}")
    raise e

# Request schemas
class PredictRequest(BaseModel):
    symptoms: List[str]
    model_type: Optional[str] = "xgboost"

class PredictTopKRequest(BaseModel):
    symptoms: List[str]
    k: Optional[int] = 3
    model_type: Optional[str] = "xgboost"

class ChatRequest(BaseModel):
    user_text: str
    session_id: Optional[str] = "default"
    openai_api_key: Optional[str] = None
    biomarkers: Optional[Dict[str, Optional[float]]] = None

@app.get("/health")
def health_check():
    logger.info("Health check endpoint accessed.")
    return {
        "status": "healthy",
        "models_loaded": {
            "random_forest": os.path.exists(ml_engine.rf_path),
            "xgboost": os.path.exists(ml_engine.xgb_path)
        }
    }

@app.post("/predict")
def predict_disease(req: PredictRequest):
    logger.info(f"Predict endpoint called with symptoms: {req.symptoms}")
    try:
        # Create symptom vector
        symptom_vector = [0] * len(hybrid_agent.symptoms)
        found_symptoms = []
        for sym in req.symptoms:
            cleaned_sym = sym.strip().replace(" ", "_").lower()
            if cleaned_sym in hybrid_agent.symptoms:
                idx = hybrid_agent.symptoms.index(cleaned_sym)
                symptom_vector[idx] = 1
                found_symptoms.append(cleaned_sym)
                
        if not found_symptoms:
            raise HTTPException(status_code=400, detail="None of the input symptoms matched the training features list.")

        # Run prediction
        preds = ml_engine.predict_topk(symptom_vector, k=1, model_type=req.model_type)
        primary = preds[0]["disease"]
        confidence = preds[0]["confidence"]
        
        info = hybrid_agent.metadata.get(primary, {"description": "No description available.", "precautions": []})
        risk = hybrid_agent.risk_engine.assess_risk(primary, confidence, found_symptoms)
        
        return {
            "predicted_disease": primary,
            "confidence": confidence,
            "description": info.get("description"),
            "precautions": info.get("precautions"),
            "risk_assessment": risk,
            "symptoms_matched": found_symptoms
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in /predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_topk")
def predict_topk_diseases(req: PredictTopKRequest):
    logger.info(f"Predict top-k endpoint called with symptoms: {req.symptoms}, k={req.k}")
    try:
        symptom_vector = [0] * len(hybrid_agent.symptoms)
        for sym in req.symptoms:
            cleaned_sym = sym.strip().replace(" ", "_").lower()
            if cleaned_sym in hybrid_agent.symptoms:
                idx = hybrid_agent.symptoms.index(cleaned_sym)
                symptom_vector[idx] = 1
                
        preds = ml_engine.predict_topk(symptom_vector, k=req.k, model_type=req.model_type)
        return {"predictions": preds}
    except Exception as e:
        logger.error(f"Error in /predict_topk: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_report")
async def analyze_medical_report(file: UploadFile = File(...)):
    logger.info(f"Report analyzer endpoint called with file: {file.filename}")
    temp_dir = r"C:\Users\DELL\Documents\AntigravityProjects\AI-Agent-Personalized-Treatment-Recommendation\temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_ext = os.path.splitext(file.filename)[1].lower()
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}{file_ext}")
    
    try:
        # Save upload to temporary path
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        summary = report_analyzer.generate_report_summary(temp_file_path)
        logger.info(f"Report analyzed successfully: {summary['status']}")
        return summary
    except Exception as e:
        logger.error(f"Error in /analyze_report: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as ex:
                logger.warning(f"Failed to remove temp file {temp_file_path}: {ex}")

@app.post("/chat")
def chat_agent(req: ChatRequest):
    logger.info(f"Chat endpoint called. Session: {req.session_id}, text length: {len(req.user_text)}")
    try:
        response_dict = hybrid_agent.predict_and_respond(
            user_text=req.user_text,
            session_id=req.session_id,
            openai_api_key=req.openai_api_key,
            biomarkers=req.biomarkers
        )
        return response_dict
    except Exception as e:
        logger.error(f"Error in /chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
