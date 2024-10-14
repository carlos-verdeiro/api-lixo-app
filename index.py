from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

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
    # Verifica se a requisição contém um arquivo
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado.'}), 400
    
    # Salvar o arquivo temporariamente
    img_path = os.path.join('temp', file.filename)
    os.makedirs('temp', exist_ok=True)  # Cria a pasta temp se não existir
    file.save(img_path)

    # Processa a imagem
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona a dimensão do batch
    
    # Classificar a imagem
    resultado = classificar_imagem(img_array)

    # Remover a imagem temporária
    os.remove(img_path)

    return jsonify({'classe': resultado})

if __name__ == '__main__':
    app.run(debug=True)
