import numpy as np
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization

# Definição do modelo
def build_vggnet(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D())
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D())
    model.add(Dropout(0.5))

    model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D())
    model.add(Dropout(0.4))

    model.add(Flatten())
    
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    return model

# Parâmetros do modelo
input_shape = (48, 48, 1)
num_classes = 4

# Criação do modelo
model = build_vggnet(input_shape, num_classes)

# Carregar os pesos do modelo
model.load_weights('saved_models/vggnet_emotions.h5')

# Função para prever a emoção de uma imagem
def predict_emotion(image_path, model, class_labels):
    img = load_img(image_path, color_mode='grayscale', target_size=(48, 48))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    emotion_index = np.argmax(predictions)
    
    return class_labels[emotion_index], predictions

# Obtenção dos rótulos das classes do treinamento
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_labels = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]


# Abrir um arquivo de texto para escrever as previsões
with open('emotion_predictions.txt', 'w') as file:
    for i in range (1000,2000):
        image_path = "training/sad/ffhq_" + str(i) + ".jpg"
        #image_path = "training/neutral/ffhq_" + str(i) + ".png"
        #image_path = "training/happy/ffhq_" + str(i) + ".png"
        #image_path = "training/angry/ffhq_" + str(i) + ".jpg"
        emotion_label, predictions = predict_emotion(image_path, model, class_labels)
        file.write(f'{emotion_label}\n')

print("Previsões salvas em 'emotion_predictions.txt'")
