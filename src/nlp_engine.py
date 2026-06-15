import re
import os

try:
    import spacy
except ImportError:
    spacy = None

class NLPEngine:
    def __init__(self, models_dir=r"C:\Users\DELL\Documents\AntigravityProjects\AI-Agent-Personalized-Treatment-Recommendation\models"):
        self.models_dir = models_dir
        
        # Try loading spaCy
        if spacy is not None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.nlp = spacy.blank("en")
        else:
            self.nlp = None
            
        # Load symptoms list from compiled model first to avoid joblib
        try:
            from src.compiled_model import SYMPTOMS
            self.symptoms = SYMPTOMS
        except ImportError:
            # Fallback loading via joblib if available
            symptoms_path = os.path.join(models_dir, "symptoms.joblib")
            if os.path.exists(symptoms_path):
                import joblib
                self.symptoms = joblib.load(symptoms_path)
            else:
                self.symptoms = []

        self.synonym_map = {
            "itchy": "itching", "itch": "itching", "rash": "skin_rash", "rashes": "skin_rash",
            "spots": "red_spots_over_body", "vomit": "vomiting", "vomiting": "vomiting",
            "throwing up": "vomiting", "puking": "vomiting", "nausea": "nausea",
            "sick to stomach": "nausea", "fever": "high_fever", "high temp": "high_fever",
            "temperature": "high_fever", "cough": "cough", "coughing": "cough",
            "cold": "continuous_sneezing", "sneezing": "continuous_sneezing", "sneeze": "continuous_sneezing",
            "shiver": "shivering", "shivering": "shivering", "chills": "chills", "chill": "chills",
            "joint pain": "joint_pain", "aching joints": "joint_pain", "stomach pain": "stomach_pain",
            "stomach ache": "stomach_pain", "belly ache": "stomach_pain", "belly pain": "belly_pain",
            "headache": "headache", "head ache": "headache", "migraine": "headache", "fatigue": "fatigue",
            "tired": "fatigue", "tiredness": "fatigue", "weakness": "fatigue", "lethargy": "lethargy",
            "weight loss": "weight_loss", "losing weight": "weight_loss", "weight gain": "weight_gain",
            "gaining weight": "weight_gain", "anxiety": "anxiety", "anxious": "anxiety",
            "indigestion": "indigestion", "heartburn": "acidity", "acidity": "acidity",
            "yellow skin": "yellowish_skin", "jaundice": "yellowish_skin", "loss of appetite": "loss_of_appetite",
            "not hungry": "loss_of_appetite", "poor appetite": "loss_of_appetite", "chest pain": "chest_pain",
            "breathlessness": "breathlessness", "breathless": "breathlessness", "short of breath": "breathlessness",
            "difficulty breathing": "breathlessness", "dizziness": "dizziness", "dizzy": "dizziness",
            "lightheaded": "dizziness", "constipation": "constipation", "diarrhea": "diarrhoea",
            "diarrhoea": "diarrhoea", "loose stools": "diarrhoea", "sweating": "sweating", "sweat": "sweating",
            "muscle pain": "muscle_pain", "muscle ache": "muscle_pain", "sore muscles": "muscle_pain",
        }

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s\(\)]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract_symptoms(self, text):
        cleaned_text = self.clean_text(text)
        
        # Word extraction with optional spaCy lemmatizer
        if self.nlp is not None:
            doc = self.nlp(cleaned_text)
            lemmas = [token.lemma_ for token in doc]
            lemma_text = " ".join(lemmas)
        else:
            # Custom simple stemmer fallback
            words = cleaned_text.split()
            lemmas = []
            for w in words:
                if w.endswith("s") and len(w) > 3:
                    lemmas.append(w[:-1])
                else:
                    lemmas.append(w)
            lemma_text = " ".join(lemmas)
            
        matched_symptoms = set()
        
        for key, val in self.synonym_map.items():
            if key in cleaned_text or key in lemma_text:
                matched_symptoms.add(val)
                
        for symptom in self.symptoms:
            cleaned_sym = symptom.replace("_", " ")
            if cleaned_sym in cleaned_text or cleaned_sym in lemma_text:
                matched_symptoms.add(symptom)
                
        symptom_vector = {}
        for symptom in self.symptoms:
            symptom_vector[symptom] = 1 if symptom in matched_symptoms else 0
            
        return {
            "symptoms_found": list(matched_symptoms),
            "vector": symptom_vector
        }
