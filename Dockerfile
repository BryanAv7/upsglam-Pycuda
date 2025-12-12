# Imagen base con CUDA y Ubuntu
FROM nvidia/cuda:13.0.1-devel-ubuntu22.04

# Evitar preguntas interactivas
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema y Python
RUN apt-get update -qq && \
    apt-get install -y python3-pip python3-dev build-essential && \
    pip3 install --upgrade pip setuptools wheel

# Crear directorio de la app
WORKDIR /app

# Copiar todo el código al contenedor
COPY . /app

# Instalar librerías de Python necesarias
RUN pip3 install flask numpy pillow pycuda opencv-python

# Exponer puerto donde corre Flask
EXPOSE 5000

# Comando para ejecutar la app principal
CMD ["python3", "main.py"]
