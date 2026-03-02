import json
import numpy as np
import logging
import sys
import os

# Configuramos el sistema de logs para que imprima mensajes en la consola del servidor
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def compute_drift_metrics(output_dir="."):
    logging.info("Iniciando escaneo de Data Drift (Métricas PSI y KS)...")

    # 1. Simulación de los resultados de los sensores
    psi = np.random.uniform(0.1, 0.4)
    ks = np.random.uniform(0.1, 0.4)
    drift_detected = bool(psi > 0.25 or ks > 0.3)

    result = {
        "psi": round(psi, 3),
        "ks": round(ks, 3),
        "drift_detected": drift_detected
    }

    # 2. Generación de Alertas (El script "grita" si hay problemas)
    if drift_detected:
        logging.warning(f"¡ALERTA DE DRIFT! Las métricas superan el umbral seguro. PSI: {result['psi']}, KS: {result['ks']}.")
        logging.warning("Acción recomendada: Evaluar reentrenamiento del modelo.")
    else:
        logging.info(f"Métricas estables. PSI: {result['psi']}, KS: {result['ks']}. No se requiere acción.")

    # 3. Escritura segura del reporte
    output_path = os.path.join(output_dir, "drift_report.json")
    try:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logging.info(f"Reporte de monitoreo guardado exitosamente en: {output_path}")
    except Exception as e:
        logging.error(f"Fallo crítico al intentar escribir el archivo de reporte: {e}")
        # En ML Ops, si un script de seguridad falla al guardar, debemos abortar el proceso
        sys.exit(1)

if __name__ == "__main__":
    # Aseguramos que el script se ejecute en el directorio actual
    compute_drift_metrics()