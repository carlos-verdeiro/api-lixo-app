from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import base64
from io import BytesIO
from flask_cors import CORS

app = Flask(__name__)

CORS(app)  # Habilita o CORS em todas as rotas

# Carregar o modelo salvo
modelo_carregado = tf.keras.models.load_model('modelo_identificacao.keras')

# Função para classificar imagem
def classificar_imagem(img_array):
    # Definindo as classes (ordem deve coincidir com o treinamento)
    classes = ['Celular', 'Pilha', 'Teclado']
    
    # Fazendo a predição
    prediction = modelo_carregado.predict(img_array)
    
    # Retorna a classe com a maior probabilidade
    return classes[np.argmax(prediction)]

@app.route('/classificar', methods=['POST'])
def classificar():
    data = request.get_json()

    # Verifica se a requisição contém a imagem
    if 'inputs' not in data or len(data['inputs']) == 0 or 'data' not in data['inputs'][0]:
        return jsonify({'error': 'Nenhuma imagem enviada.'}), 400
    
    base64_image = data['inputs'][0]['data']['image']['base64']

    # Decodifica a imagem base64
    try:
        img_data = base64.b64decode(base64_image)
    except Exception as e:
        return jsonify({'error': 'Falha ao decodificar a imagem.'}), 400

    # Carrega a imagem em um formato utilizável
    try:
        img = image.load_img(BytesIO(img_data), target_size=(128, 128))
    except Exception as e:
        return jsonify({'error': 'Erro ao carregar a imagem.'}), 400

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona a dimensão do batch
    
    # Classificar a imagem
    resultado = classificar_imagem(img_array)

    return jsonify({'classe': resultado})

if __name__ == '__main__':
    app.run(debug=True)
