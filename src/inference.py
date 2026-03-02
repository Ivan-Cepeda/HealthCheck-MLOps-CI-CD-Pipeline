import pandas as pd
import pickle
import os
import logging

# Configuramos un registro básico para saber qué ocurre en producción----
logging.basicConfig(level=logging.INFO)

# PASO A: Cargar el modelo de forma global
MODEL_PATH = "model/health_risk_model.pkl"

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo en la ruta: {MODEL_PATH}")
        
    with open(MODEL_PATH, "rb") as f:
        # El modelo vive en memoria de forma persistente
        model = pickle.load(f)
        logging.info("Modelo cargado exitosamente en memoria.")
        
except Exception as e:
    logging.error(f"Error crítico al cargar el modelo: {e}")
    model = None # Evita que el programa falle silenciosamente


# PASO B:
def predict_risk(data: dict) -> dict:
    # 1. Validación de seguridad inicial
    if model is None:
        return {"error": "El modelo no está disponible en el servidor."}
    
    if not data:
        return {"error": "No se proporcionaron datos para la predicción."}
    
    # Si la edad viene en los datos y es menor a 0, lanzamos el ValueError que el test espera
    if "age" in data and data["age"] < 0:
        raise ValueError("Invalid input: 'age' cannot be a negative value.")

    try:
        # 2. Transformación de datos
        df = pd.DataFrame([data])
        
        # 3. Predicción
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        confidence = probabilities[prediction]
        
        # 4. Retorno estructurado
        return {
            "risk": bool(prediction), 
            "confidence": float(confidence)
        }
        
    except Exception as e:
        # Capturamos errores inesperados (ej. faltan columnas en el diccionario)
        logging.error(f"Error durante la inferencia: {e}")
        return {"error": str(e)}