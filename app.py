from flask import Flask, request, jsonify, render_template
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import os
from io import BytesIO
from PIL import Image
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch


app = Flask(__name__)

# URL del modelo a descargar
url = 'https://drive.google.com/drive/folders/1iwNmwItnO1HZttp2cAJL1P7UMCbMyABV'
# Ruta del modelo descargado
model_sentiment_path = 'models/sentiment_model.safetensors' 

# Descargamos el modelo si no existe
if not os.path.exists(model_sentiment_path):
    print("Descargando el archivo del modelo...")
    r = requests.get(url)
    with open(model_sentiment_path, 'wb') as f:
        f.write(r.content)
    print("Archivo descargado exitosamente.")

# Ruta del modelo de clasificación de imágenes
model_cnn_path = 'models/cellphone_screen.keras'

#Cargar el tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('models/')

# Verificar si el archivo existe
if os.path.exists(model_sentiment_path):
    sentiment_model = DistilBertForSequenceClassification.from_pretrained(model_sentiment_path)
    print("Modelo sentiment_analysis_model cargado exitosamente.")
else:
    print("El modelo sentiment_model no se encuentra en la ruta especificada.")

if os.path.exists(model_cnn_path):
    cnn_model = load_model(model_cnn_path)
    print("Modelo cnn_model cargado exitosamente.")
else:
    print("El modelo cnn_model no se encuentra en la ruta especificada.")

#Ruta principal HOME, cuando se ingresa a la aplicación
@app.route('/')
def home():
    return render_template("index.html")

# Rutas de la API
#Ruta para el análisis de sentimientos 
#Revisar los metodos http que van a recibir
@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.json.get('review')
        if data is None:
            return jsonify({'error': 'No review provided'}), 400
        
        # Tokenización y preparación del input
        inputs = tokenizer(data, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Realiza la predicción
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()  # Obtener la clase con la puntuación más alta

        return jsonify({'sentiment': 'positive' if prediction == 1 else 'negative'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        result = 'not broken' if prediction[0] > 0.5 else 'broken'

        return jsonify({'screen_status': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Correr la aplicación
if __name__ == '__main__':
    app.run(debug=True)
