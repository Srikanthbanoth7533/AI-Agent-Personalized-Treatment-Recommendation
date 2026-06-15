import os
try:
    import joblib
except ImportError:
    joblib = None
from openai import OpenAI
from src.nlp_engine import NLPEngine
from src.ml_engine import MLEngine
from src.risk_engine import RiskEngine

class HybridAgent:
    def __init__(self, models_dir=r"C:\Users\DELL\Documents\AntigravityProjects\AI-Agent-Personalized-Treatment-Recommendation\models"):
        self.models_dir = models_dir
        self.nlp_engine = NLPEngine(models_dir)
        self.ml_engine = MLEngine(models_dir)
        self.risk_engine = RiskEngine()
        
        # Try loading compiled model lists first to avoid joblib
        try:
            from src.compiled_model import SYMPTOMS, DISEASE_METADATA
            self.symptoms = SYMPTOMS
            self.metadata = DISEASE_METADATA
        except ImportError:
            self.symptoms = []
            self.metadata = {}

        # Fallback to joblib if available
        if (not self.symptoms or not self.metadata) and joblib is not None:
            try:
                symptoms_path = os.path.join(models_dir, "symptoms.joblib")
                if os.path.exists(symptoms_path):
                    self.symptoms = joblib.load(symptoms_path)
                
                metadata_path = os.path.join(models_dir, "disease_metadata.joblib")
                if os.path.exists(metadata_path):
                    self.metadata = joblib.load(metadata_path)
            except Exception as e:
                print(f"Failed to load joblib configuration: {e}")
            
        # Memory store: {session_id: list of messages}
        self.memory = {}

    def get_history(self, session_id):
        if not session_id:
            return []
        if session_id not in self.memory:
            self.memory[session_id] = []
        return self.memory[session_id]

    def add_to_history(self, session_id, role, content):
        if not session_id:
            return
        history = self.get_history(session_id)
        history.append({"role": role, "content": content})
        # Keep only last 10 messages to save context/memory size
        if len(history) > 10:
            history.pop(0)

    def predict_and_respond(self, user_text, session_id="default", openai_api_key=None, biomarkers=None):
        history = self.get_history(session_id)
        
        # Extract symptoms
        nlp_res = self.nlp_engine.extract_symptoms(user_text)
        symptoms_found = nlp_res["symptoms_found"]
        
        # If no symptoms are found, check if it's just conversational
        if not symptoms_found and not biomarkers:
            response = self.handle_conversational_chat(user_text, history, openai_api_key)
            self.add_to_history(session_id, "user", user_text)
            self.add_to_history(session_id, "assistant", response)
            return {
                "response": response,
                "symptoms_found": [],
                "predictions": [],
                "risk_assessment": {"level": "Low", "reasons": ["No symptoms or biomarkers reported."]},
                "type": "chat"
            }

        # Otherwise, we have symptoms and/or biomarkers. Map symptoms to vector.
        symptom_vector = [nlp_res["vector"].get(s, 0) for s in self.symptoms]
        
        # Check if we have symptoms in vector (at least one 1)
        has_symptoms = sum(symptom_vector) > 0
        
        predictions = []
        primary_disease = None
        confidence = 0.0
        disease_info = {"description": "No specific information available.", "precautions": []}
        
        if has_symptoms:
            try:
                predictions = self.ml_engine.predict_topk(symptom_vector, k=3)
                primary_disease = predictions[0]["disease"]
                confidence = predictions[0]["confidence"]
                disease_info = self.metadata.get(primary_disease, {
                    "description": "No specific description available.", 
                    "precautions": ["Consult a medical practitioner."]
                })
            except Exception as e:
                print(f"ML Prediction failed: {e}")
                primary_disease = "Unknown condition"
                predictions = [{"disease": "Unknown", "confidence": 1.0}]
        else:
            primary_disease = "No symptoms matched"
            predictions = [{"disease": "No symptoms matched", "confidence": 1.0}]

        # Assess Risk
        risk_res = self.risk_engine.assess_risk(
            primary_disease if has_symptoms else "None", 
            confidence, 
            symptoms_found, 
            biomarkers
        )
        
        # Formulate response
        if openai_api_key:
            response = self.generate_genai_response(
                user_text, history, symptoms_found, predictions, disease_info, risk_res, biomarkers, openai_api_key
            )
        else:
            response = self.generate_rule_response(
                symptoms_found, predictions, disease_info, risk_res, biomarkers
            )
            
        self.add_to_history(session_id, "user", user_text)
        self.add_to_history(session_id, "assistant", response)
        
        return {
            "response": response,
            "symptoms_found": symptoms_found,
            "predictions": predictions,
            "risk_assessment": risk_res,
            "type": "diagnosis"
        }

    def handle_conversational_chat(self, text, history, api_key):
        if api_key:
            try:
                client = OpenAI(api_key=api_key)
                messages = [
                    {"role": "system", "content": "You are a professional, friendly, and helpful AI healthcare assistant. Welcoming the patient, answering general health questions, or asking them to describe their symptoms so we can analyze them. Remember to always display a professional disclaimer that you are an AI, not a doctor."}
                ]
                messages.extend(history)
                messages.append({"role": "user", "content": text})
                
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=250,
                    temperature=0.7
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"OpenAI conversational chat failed: {e}")
                
        return ("Hello! I am your personalized AI Healthcare Assistant. "
                "I am here to help analyze symptoms, review medical reports (PDFs/Images), and provide risk assessments. "
                "Please describe the symptoms you are experiencing (e.g. 'I have a headache and muscle pain') or upload a medical report to get started.\n\n"
                "*Disclaimer: I am an AI assistant, not a doctor. Please consult a healthcare professional for actual medical diagnosis and treatment.*")

    def generate_genai_response(self, text, history, symptoms, predictions, disease_info, risk, biomarkers, api_key):
        try:
            client = OpenAI(api_key=api_key)
            
            prompt = f"""
            User input: "{text}"
            Symptoms matched: {symptoms}
            Top predictions: {predictions}
            Primary Disease Description: {disease_info.get('description')}
            Recommended Precautions: {disease_info.get('precautions')}
            Risk Level: {risk['level']}
            Risk Reasons: {risk['reasons']}
            Biomarkers parsed: {biomarkers}
            
            Please formulate an empathetic, clear, and professional response that:
            1. Summarizes the findings (predicted diseases & confidence scores).
            2. Explains the symptoms matched.
            3. Details the primary predicted condition and its typical description.
            4. Recommends the corresponding precautions.
            5. Outlines the risk level assessment.
            6. Boldly states a medical disclaimer.
            """
            
            messages = [
                {"role": "system", "content": "You are an expert AI clinical agent providing personalized health guidance. Format your response beautifully using markdown."}
            ]
            messages.extend(history)
            messages.append({"role": "user", "content": prompt})
            
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=600,
                temperature=0.3
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"OpenAI diagnosis prompt failed: {e}. Falling back to rule-based response.")
            return self.generate_rule_response(symptoms, predictions, disease_info, risk, biomarkers)

    def generate_rule_response(self, symptoms, predictions, disease_info, risk, biomarkers):
        resp = "### AI Clinical Assessment\n\n"
        
        if symptoms:
            resp += f"**Symptoms Detected:** {', '.join(symptoms)}\n\n"
            
        if biomarkers:
            parsed = []
            for k, v in biomarkers.items():
                if v is not None:
                    parsed.append(f"{k.capitalize()}: {v}")
            if parsed:
                resp += f"**Biomarkers Extracted:** {', '.join(parsed)}\n\n"

        if predictions:
            resp += "**Top Disease Predictions:**\n"
            for pred in predictions:
                resp += f"- **{pred['disease']}** (Confidence: {pred['confidence'] * 100:.1f}%)\n"
            resp += "\n"

            primary = predictions[0]['disease']
            resp += f"#### About {primary}:\n"
            resp += f"{disease_info.get('description', 'No description available.')}\n\n"
            
            precautions = disease_info.get('precautions', [])
            if precautions:
                resp += "#### Recommended Precautions:\n"
                for prec in precautions:
                    resp += f"- {prec}\n"
                resp += "\n"

        resp += f"#### Risk Assessment: **{risk['level']}**\n"
        for reason in risk["reasons"]:
            resp += f"- {reason}\n"
        resp += "\n"
        
        resp += "---\n"
        resp += "***Disclaimer:** I am an AI agent, not a qualified doctor. The predictions and advice generated are for educational purposes. If you are experiencing severe symptoms, please visit a clinic or hospital immediately.*"
        
        return resp

if __name__ == "__main__":
    agent = HybridAgent()
    res = agent.predict_and_respond("I have a high fever and headache.")
    print(res["response"])
