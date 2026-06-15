import pytest
from src.nlp_engine import NLPEngine

def test_symptom_extraction_direct():
    engine = NLPEngine()
    text = "I am experiencing itching and a skin rash."
    res = engine.extract_symptoms(text)
    
    assert "itching" in res["symptoms_found"]
    assert "skin_rash" in res["symptoms_found"]

def test_symptom_extraction_synonyms():
    engine = NLPEngine()
    text = "I have a high temp, headache, and feel like vomiting."
    res = engine.extract_symptoms(text)
    
    assert "high_fever" in res["symptoms_found"]
    assert "headache" in res["symptoms_found"]
    assert "vomiting" in res["symptoms_found"]

def test_symptom_vector_structure():
    engine = NLPEngine()
    text = "Nothing hurts, feeling completely fine."
    res = engine.extract_symptoms(text)
    
    assert len(res["symptoms_found"]) == 0
    assert len(res["vector"]) == len(engine.symptoms)
    assert sum(res["vector"].values()) == 0
