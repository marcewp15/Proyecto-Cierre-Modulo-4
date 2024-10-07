from flask import Flask, request, jsonify
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image
import os

app = Flask(__name__)

# Ruta del modelo
model_sentiment_path = 'models/sentiment_model.h5'  # Cuando esten listos los modelos, colocarlos en la carpeta models
model_cnn_path = 'models/cnn_model.h5'  

# Variables para simular los modelos
sentiment_model = None  # Simulación
cnn_model = None  # Simulación

# Verificar si el archivo existe PRUEBA
if os.path.exists(model_sentiment_path):
    print("Modelo sentiment_model cargado exitosamente.")
else:
    print("El modelo sentiment_model no se encuentra en la ruta especificada.")

if os.path.exists(model_cnn_path):
    print("Modelo cnn_model cargado exitosamente.")
else:
    print("El modelo cnn_model no se encuentra en la ruta especificada.")

#Ruta principal HOME, cuando se ingresa a la aplicación
@app.route('/')
def home():
    return "API de análisis de sentimiento y clasificación de imágenes. /nUsa /analyze_sentiment y /classify_image."

# Rutas de la API
#Ruta para el análisis de sentimientos 
#Revisar los metodos http que van a recibir
@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.json['review']
    # Simular preprocesamiento y predicción
    # Revisar la lógica según lo que necesitemos
    if "good" in data.lower():
        prediction = np.array([[0.9]])  # Simular predicción positiva
    else:
        prediction = np.array([[0.1]])  # Simular predicción negativa

    return jsonify({'sentiment': 'positive' if prediction > 0.5 else 'negative'}) #convertir un diccionario de Python en una respuesta en formato JSON

#Ruta para la clasificación de imagenes
#Al igual que el anterior, revisar los metoos http que va a recibir
@app.route('/classify_image', methods=['POST'])
def classify_image():
    img = request.files['image']
    # Simular preprocesamiento de imagen
    # Solo se devuelve un resultado
    prediction = np.array([[0.7]])  # Simular predicción

    return jsonify({'screen_status': 'broken' if prediction > 0.5 else 'not broken'}) #convertir un diccionario de Python en una respuesta en formato JSON, con la respuesta que se necesita

# Correr la aplicación
if __name__ == '__main__':
    app.run(debug=True)
