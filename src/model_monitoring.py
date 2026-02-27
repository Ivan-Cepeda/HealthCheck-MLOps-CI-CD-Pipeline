
import json
import numpy as np

def compute_drift_metrics():
    # Simula resultados de PSI y KS
    psi = np.random.uniform(0.1, 0.4)
    ks = np.random.uniform(0.1, 0.4)
    drift_detected = psi > 0.25 or ks > 0.3

    result = {
        "psi": round(psi, 3),
        "ks": round(ks, 3),
        "drift_detected": drift_detected
    }

    with open("drift_report.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    compute_drift_metrics()
