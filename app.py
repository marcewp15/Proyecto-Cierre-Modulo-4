from flask import Flask, request, jsonify
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import os
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Ruta del modelo
#model_sentiment_path = 'models/sentiment_analysis_model.pkl'  
model_cnn_path = 'models/cellphone_screen.keras'  

# Verificar si el archivo existe PRUEBA
#if os.path.exists(model_sentiment_path):
#    sentiment_model = load_model(model_sentiment_path)
#    print("Modelo sentiment_analysis_model cargado exitosamente.")
#else:
#    print("El modelo sentiment_model no se encuentra en la ruta especificada.")

if os.path.exists(model_cnn_path):
    cnn_model = load_model(model_cnn_path)
    print("Modelo cnn_model cargado exitosamente.")
else:
    print("El modelo cnn_model no se encuentra en la ruta especificada.")

#Ruta principal HOME, cuando se ingresa a la aplicación
@app.route('/')
def home():
    return "API de análisis de sentimiento y clasificación de imágenes. Usa /analyze_sentiment y /classify_image."

# Rutas de la API
#Ruta para el análisis de sentimientos 
#Revisar los metodos http que van a recibir
#@app.route('/analyze_sentiment', methods=['POST'])
#def analyze_sentiment():
#    data = request.json['review']
    # Simular preprocesamiento y predicción
    # Revisar la lógica según lo que necesitemos
#    if "good" in data.lower():
#        prediction = np.array([[0.9]])  # Simular predicción positiva
#    else:
#        prediction = np.array([[0.1]])  # Simular predicción negativa

#    return jsonify({'sentiment': 'positive' if prediction > 0.5 else 'negative'}) #convertir un diccionario de Python en una respuesta en formato JSON

#Ruta para la clasificación de imagenes
#Al igual que el anterior, revisar los metoos http que va a recibir
@app.route('/classify_image', methods=['POST'])
def classify_image():
    # Verifica si se ha subido una imagen
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    img_file = request.files['image']
    
    try:
        # Convertir el archivo en BytesIO y abrirlo con PIL
        img = Image.open(BytesIO(img_file.read()))  # Usamos PIL para manejar la imagen
        img = img.resize((150, 150))  # Ajusta el tamaño según lo que el modelo espera

        # Convertir a array de numpy y agregar una dimensión para el batch
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Añade una dimensión para el batch

        # Normalizar la imagen
        img_array /= 255.0  # Normaliza si es necesario (según lo entrenado)

        # Realiza la predicción
        prediction = cnn_model.predict(img_array)

        # Devuelve la predicción
        result = 'broken' if prediction[0] > 0.5 else 'not broken'

        return jsonify({'screen_status': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Correr la aplicación
if __name__ == '__main__':
    app.run(debug=True)
