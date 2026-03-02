import pytest
from inference import predict_risk

#Función de test
def test_predict_risk():
    """
    Test que valida que el modelo responde correctamente
    con la estructura esperada.
    """
    sample_input = {
        "age": 35,
        "gender": "male",
        "smoker": "no",
        "bmi": 26.7
    }

    result = predict_risk(sample_input)

    # Validaciones del contrato de datos
    assert "risk" in result, "La respuesta debe contener la clave 'risk'"
    assert isinstance(result["risk"], bool), "El riesgo debe ser un booleano"
    assert "confidence" in result, "La respuesta debe incluir confianza"
    assert 0 <= result["confidence"] <= 1, "Confianza entre 0 y 1"

def test_predict_risk_invalid_input():
    """
    Test que verifica el manejo de inputs inválidos
    """
    invalid_input = {"age": -5}

    # Esta es la forma profesional en Pytest de asegurar que una función falle
    with pytest.raises(ValueError) as excinfo:
        predict_risk(invalid_input)
    
    # Opcional: Validar que el mensaje de error sea el correcto
    assert "age" in str(excinfo.value).lower()