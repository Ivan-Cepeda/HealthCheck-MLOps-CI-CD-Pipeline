import pandas as pd
import pickle

def predict_risk(data: dict):
    with open("model/health_risk_model.pkl", "rb") as f:
        model = pickle.load(f)
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    confidence = probabilities[prediction]
    return {"risk": bool(prediction), "confidence": float(confidence)}