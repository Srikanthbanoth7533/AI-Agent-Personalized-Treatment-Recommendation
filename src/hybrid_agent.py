import os
try:
    import joblib
except ImportError:
    joblib = None
from openai import OpenAI
from src.nlp_engine import NLPEngine
from src.ml_engine import MLEngine
from src.risk_engine import RiskEngine

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

class HybridAgent:
    def __init__(self, models_dir=None):
        self.models_dir = models_dir or DEFAULT_MODELS_DIR
        self.nlp_engine = NLPEngine(self.models_dir)
        self.ml_engine = MLEngine(self.models_dir)
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

    def is_conversational_query(self, text):
        text_lower = text.lower().strip()
        conversational_keywords = [
            "hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening",
            "thank you", "thanks", "awesome", "great", "perfect", "cool",
            "who are you", "what can you do", "help", "capabilities", "features", "functions",
            "how are you", "what's up", "hey there"
        ]
        for kw in conversational_keywords:
            if kw in text_lower or text_lower.startswith(kw):
                return True
        return False

    def predict_and_respond(self, user_text, session_id="default", openai_api_key=None, biomarkers=None):
        history = self.get_history(session_id)
        
        # Extract symptoms from the current message
        nlp_res = self.nlp_engine.extract_symptoms(user_text)
        current_symptoms = nlp_res["symptoms_found"]
        
        # Determine if we should route to conversational chat
        is_conv = self.is_conversational_query(user_text)
        
        # Check if there are any symptoms in the history
        has_history_symptoms = False
        for msg in history:
            if msg["role"] == "user":
                prev_res = self.nlp_engine.extract_symptoms(msg["content"])
                if prev_res["symptoms_found"]:
                    has_history_symptoms = True
                    break
                    
        if (not current_symptoms and is_conv) or (not current_symptoms and not has_history_symptoms and not biomarkers):
            response = self.handle_conversational_chat(user_text, history, openai_api_key, biomarkers)
            self.add_to_history(session_id, "user", user_text)
            self.add_to_history(session_id, "assistant", response)
            return {
                "response": response,
                "symptoms_found": [],
                "predictions": [],
                "risk_assessment": {"level": "Low", "reasons": ["No symptoms or biomarkers reported."]},
                "type": "chat"
            }

        # Otherwise, this is a diagnosis query. Accumulate symptoms from history and current message.
        accumulated_symptoms = set(current_symptoms)
        for msg in history:
            if msg["role"] == "user":
                prev_res = self.nlp_engine.extract_symptoms(msg["content"])
                accumulated_symptoms.update(prev_res["symptoms_found"])
                
        symptoms_found = list(accumulated_symptoms)

        # Map symptoms to vector.
        symptom_vector = [1 if s in symptoms_found else 0 for s in self.symptoms]
        
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
                symptoms_found, predictions, disease_info, risk_res, biomarkers, history
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

    def handle_conversational_chat(self, text, history, api_key, biomarkers=None):
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
                
        # Rule-based dynamic conversational handler when OpenAI API Key is missing
        text_lower = text.lower().strip()
        
        # Check greetings
        if any(g in text_lower for g in ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]):
            resp = ("Hello! I'm Aegis AI, your personalized healthcare assistant. How can I help you today? "
                    "You can describe your symptoms (e.g. 'I have a fever and chills'), ask general health questions, "
                    "or upload a medical report for analysis.\n\n")
            if biomarkers:
                parsed = [f"{k.capitalize()}: {v}" for k, v in biomarkers.items() if v is not None]
                if parsed:
                    resp += f"*System Note: I have loaded your medical report values ({', '.join(parsed)}). Let me know if you would like me to analyze them alongside your symptoms.* \n\n"
            resp += "*Disclaimer: I am an AI, not a doctor. For any medical emergency, please consult a physician.*"
            return resp
                    
        # Check capabilities / who are you / help
        if any(keyword in text_lower for keyword in ["who are you", "what can you do", "help", "capabilities", "features", "functions"]):
            return ("I am Aegis AI, a clinical decision support and personalized health assistant. "
                    "My core capabilities include:\n"
                    "1. **Symptom Extraction & Diagnosis:** Analyzing symptoms you describe and predicting potential diseases using machine learning.\n"
                    "2. **Risk Assessment:** Determining severity levels (Low/Medium/High) based on symptoms and conditions.\n"
                    "3. **Report Analysis:** Parsing uploaded medical reports (PDF/Images) to extract key lab biomarkers like Glucose, Hemoglobin, and Cholesterol.\n"
                    "4. **Precautions & Descriptions:** Providing detailed precautions and info for matched diseases.\n\n"
                    "How can I assist you with your health query today?\n\n"
                    "*Disclaimer: I am an AI, not a doctor. For any medical emergency, please consult a physician.*")
                    
        # Check appreciation
        if any(keyword in text_lower for keyword in ["thank you", "thanks", "awesome", "great", "perfect", "cool"]):
            return ("You're welcome! I'm happy to help. Let me know if you have any other symptoms or medical reports you'd like me to analyze.\n\n"
                    "*Disclaimer: I am an AI assistant. Please seek professional medical advice for diagnoses.*")
                    
        # General response helper
        return (f"I received your message: \"{text}\". I couldn't find any symptoms or biomarkers in it to run a diagnostic prediction. "
                "Could you please describe any symptoms you are experiencing (like fever, cough, joint pain, or headache) "
                "or upload a medical report? This will allow me to provide a clinical risk assessment.\n\n"
                "*Disclaimer: I am an AI assistant. Please seek professional medical advice.*")

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

    def generate_rule_response(self, symptoms, predictions, disease_info, risk, biomarkers, history=None):
        resp = "### Aegis AI - Clinical Decision Support Analysis\n\n"
        
        if symptoms:
            clean_symptoms = [s.replace("_", " ") for s in symptoms]
            resp += f"Based on the symptoms you've reported (**{', '.join(clean_symptoms)}**), I have performed a multi-model clinical assessment.\n\n"
        else:
            resp += "I have reviewed your active session context and biomarkers.\n\n"

        if biomarkers:
            parsed = []
            for k, v in biomarkers.items():
                if v is not None:
                    parsed.append(f"**{k.capitalize()}**: {v}")
            if parsed:
                resp += f"**Extracted Lab Biomarkers:** {', '.join(parsed)}\n\n"

            # Report-Based Prediction Insights (Requirement 14)
            glucose = biomarkers.get("glucose")
            hemo = biomarkers.get("hemoglobin")
            chol = biomarkers.get("cholesterol")
            insights = []
            if glucose is not None and glucose > 100:
                insights.append("Your elevated glucose levels indicate a potential risk for hyperglycemia/diabetes. This should be monitored.")
            if hemo is not None and hemo < 12.0:
                insights.append("Your low hemoglobin levels indicate potential anemia, which can cause symptoms like fatigue or weakness.")
            if chol is not None and chol >= 200:
                insights.append("Your total cholesterol is elevated, suggesting a potential cardiovascular risk.")
            if insights:
                resp += "##### 📈 Report-Based Insights\n"
                for ins in insights:
                    resp += f"- {ins}\n"
                resp += "\n"

        if predictions:
            primary = predictions[0]['disease']
            primary_conf = predictions[0]['confidence'] * 100
            
            resp += f"#### 🩺 Differential Diagnosis\n"
            resp += f"Our classification engine suggests **{primary}** as the primary likelihood (Confidence: **{primary_conf:.1f}%**).\n"
            
            if len(predictions) > 1:
                resp += "\nComparing alternative possibilities, we also evaluated:\n"
                for pred in predictions[1:3]:
                    resp += f"- **{pred['disease']}** (Confidence: {pred['confidence'] * 100:.1f}%)\n"
                resp += f"\n*Reasoning:* **{primary}** is predicted with higher likelihood because the clinical vector of your symptoms fits its diagnostic pattern closely. "
                if len(symptoms) > 1:
                    resp += f"Specifically, the combination of symptoms like {', '.join(clean_symptoms[:2])} strongly matches the signature of {primary} over other conditions."
                resp += "\n\n"
            
            resp += f"##### About {primary}\n"
            resp += f"{disease_info.get('description', 'No specific description available.')}\n\n"
            
            precautions = disease_info.get('precautions', [])
            if precautions:
                resp += "##### 📋 Recommended Next Steps & Precautions\n"
                for prec in precautions:
                    resp += f"- {prec.capitalize()}\n"
                resp += "\n"

        # Risk Assessment (Requirement 16)
        resp += f"#### ⚠️ Risk Profile: **{risk['level']}**\n"
        for reason in risk["reasons"]:
            resp += f"- {reason}\n"
        resp += "\n"

        # Follow-Up Questions (Requirement 6)
        resp += "#### ❓ Follow-Up Inquiries\n"
        resp += "To help me refine this differential diagnosis, please consider:\n"
        resp += "1. *Onset & Duration:* How many days have you been experiencing these symptoms?\n"
        if symptoms and "high_fever" in symptoms:
            resp += "2. *Pattern:* Is the fever constant, or does it spike at specific times of the day?\n"
        else:
            resp += "2. *Additional symptoms:* Have you noticed any other changes like changes in sleep, appetite, or fatigue?\n"
        resp += "3. *Environment:* Have you recently traveled or been exposed to anyone with similar symptoms?\n\n"
        
        resp += "---\n"
        resp += "***Disclaimer:** I am an AI clinical assistant, not a doctor. The predictions and advice generated are for decision-support and educational purposes only. If you are experiencing severe symptoms, please visit a clinic or hospital immediately.*"
        
        return resp

if __name__ == "__main__":
    agent = HybridAgent()
    res = agent.predict_and_respond("I have a high fever and headache.")
    print(res["response"])
