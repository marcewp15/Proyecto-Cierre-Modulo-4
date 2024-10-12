# Usar una imagen base de Python 3.11
FROM python:3.11-slim

# Establecer el directorio de trabajo en /app
WORKDIR /app

# Copiar el archivo de requisitos a la imagen y instalar las dependencias
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código de la aplicación al contenedor
COPY . .

# Exponer el puerto que usará la aplicación (Heroku usa el puerto 5000 por defecto)
EXPOSE 5000

# Comando para iniciar la aplicación Flask usando Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
