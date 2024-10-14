from http.server import BaseHTTPRequestHandler
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        # Enviar resposta inicial
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        
        # Carregar o modelo
        model_path = os.path.join(os.path.dirname(__file__), 'modelo_identificacao.keras')
        model = tf.keras.models.load_model(model_path)
        
        # Caminho da imagem (substituir pelo método que você usará para receber imagens)
        img_path = os.path.join(os.path.dirname(__file__), 'roi.jpg')

        # Classificar imagem
        result = self.classificar_imagem(img_path, model)
        
        # Escrever o resultado na resposta HTTP
        self.wfile.write(result.encode('utf-8'))
    
    def classificar_imagem(self, img_path, model):
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Adiciona a dimensão do batch
        
        # Definindo as classes
        classes = ['Celular', 'Pilha', 'Teclado']
        
        # Fazendo a predição
        prediction = model.predict(img_array)
        
        # Retornando o resultado da predição
        return f"Esta imagem é provavelmente um: {classes[np.argmax(prediction)]}"

