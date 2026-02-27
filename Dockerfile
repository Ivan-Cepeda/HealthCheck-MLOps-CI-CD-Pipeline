# 1. Usar una imagen de Python oficial (nuestro sistema operativo base)
FROM python:3.10-slim

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiar el inventario e instalar dependencias (hacer esto primero optimiza tiempos)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar el resto del código y el modelo
COPY src/ ./src/
COPY model/ ./model/

# 5. Definir qué comando se ejecuta al encender el contenedor (ejemplo: levantar la API)
# CMD ["python", "src/inference.py"]