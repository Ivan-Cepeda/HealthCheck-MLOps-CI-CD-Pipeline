# 1. Base ligera y segura
FROM python:3.10-slim

# 2. Directorio de trabajo
WORKDIR /app

# --- FASE DE ENSAMBLAJE (Rara vez cambia) ---
# 3. Copiamos SOLO el inventario de dependencias
COPY requirements.txt .

# 4. Instalamos las herramientas. Añadimos dependencias del sistema necesarias para compilar librerías de ML si hiciera falta.
RUN pip install --no-cache-dir -r requirements.txt

# --- FASE DE PINTURA (Cambia frecuentemente) ---
# 5. Copiamos el código y el modelo al final para aprovechar la caché de Docker
COPY src/ ./src/
COPY model/ ./model/

# 6. Documentamos el puerto
EXPOSE 8000

# 7. Monitoreo de salud para garantizar que la API de predicción está viva
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# 8. Comando de encendido para producción
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]