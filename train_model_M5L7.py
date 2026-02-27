import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================
print("Cargando datos...")
df = pd.read_csv('patients.csv')

print(f"Dimensiones del dataset: {df.shape}")
print(f"\nPrimeras filas:")
print(df.head())
print(f"\nInformación del dataset:")
print(df.info())
print(f"\nDistribución de la variable objetivo:")
print(df['risk'].value_counts())

# ============================================================================
# 2. PREPARAR DATOS
# ============================================================================
# Separar características (X) y variable objetivo (y)
X = df.drop('risk', axis=1)
y = df['risk']

# Dividir en train/test (opcional, para evaluar el modelo)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDatos de entrenamiento: {X_train.shape}")
print(f"Datos de prueba: {X_test.shape}")

# ============================================================================
# 3. DEFINIR EL PIPELINE (IDÉNTICO AL .pkl)
# ============================================================================

# Definir las columnas numéricas y categóricas
numeric_features = ['age', 'bmi']
categorical_features = ['gender', 'smoker']

# Crear el preprocesador (ColumnTransformer)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Crear el clasificador (LogisticRegression con los mismos parámetros)
classifier = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='lbfgs',
    max_iter=100,
    random_state=42  # Agregamos random_state para reproducibilidad
)

# Crear el pipeline completo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

print("\n" + "="*70)
print("ESTRUCTURA DEL PIPELINE")
print("="*70)
print(pipeline)

# ============================================================================
# 4. ENTRENAR EL MODELO
# ============================================================================
print("\n" + "="*70)
print("ENTRENANDO EL MODELO...")
print("="*70)

pipeline.fit(X_train, y_train)

print("✓ Modelo entrenado exitosamente!")

# ============================================================================
# 5. EVALUAR EL MODELO
# ============================================================================
print("\n" + "="*70)
print("EVALUACIÓN DEL MODELO")
print("="*70)

# Predicciones
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

# Accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nAccuracy en entrenamiento: {train_accuracy:.4f}")
print(f"Accuracy en prueba: {test_accuracy:.4f}")

print("\n--- Matriz de Confusión (Test) ---")
print(confusion_matrix(y_test, y_pred_test))

print("\n--- Reporte de Clasificación (Test) ---")
print(classification_report(y_test, y_pred_test))

# ============================================================================
# 6. GUARDAR EL MODELO
# ============================================================================
import os
output_dir = 'outputs'
model_filename = os.path.join(output_dir, 'health_risk_model_trained.pkl')

with open(model_filename, 'wb') as f:
    pickle.dump(pipeline, f)

print(f"\n Modelo guardado como: {model_filename}")

# ============================================================================
# 7. VERIFICAR QUE EL PIPELINE ES IDÉNTICO
# ============================================================================
print("\n" + "="*70)
print("VERIFICACIÓN DE LA ESTRUCTURA DEL PIPELINE")
print("="*70)

print(f"\nNúmero de pasos: {len(pipeline.steps)}")
print(f"Nombres de pasos: {[name for name, _ in pipeline.steps]}")

preprocessor_step = pipeline.named_steps['preprocessor']
classifier_step = pipeline.named_steps['classifier']

print(f"\nPreprocesador:")
print(f"  - Tipo: {type(preprocessor_step).__name__}")
print(f"  - Transformadores: {[(name, type(trans).__name__, cols) for name, trans, cols in preprocessor_step.transformers_]}")

print(f"\nClasificador:")
print(f"  - Tipo: {type(classifier_step).__name__}")
print(f"  - C: {classifier_step.C}")
print(f"  - Penalty: {classifier_step.penalty}")
print(f"  - Solver: {classifier_step.solver}")
print(f"  - Max iter: {classifier_step.max_iter}")

print("\n" + "="*70)
print("EJEMPLO DE PREDICCIÓN")
print("="*70)

# Crear un ejemplo de paciente
ejemplo = pd.DataFrame({
    'age': [45],
    'gender': ['male'],
    'smoker': ['yes'],
    'bmi': [28.5]
})

prediccion = pipeline.predict(ejemplo)
probabilidad = pipeline.predict_proba(ejemplo)

print(f"\nPaciente ejemplo: {ejemplo.to_dict('records')[0]}")
print(f"Predicción: {'Alto riesgo' if prediccion[0] == 1 else 'Bajo riesgo'} (clase {prediccion[0]})")
print(f"Probabilidades: Clase 0: {probabilidad[0][0]:.4f}, Clase 1: {probabilidad[0][1]:.4f}")

print("\n" + "="*70)
print("¡PROCESO COMPLETADO!")
print("="*70)
