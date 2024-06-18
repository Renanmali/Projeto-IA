import tensorflow as tf
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import backend as K

# Diretório das imagens organizadas por emoções
data_dir = 'training'

# Configuração do ImageDataGenerator para pré-processamento das imagens
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% dos dados serão usados para validação
)

# Gerador de dados de treinamento
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(48, 48),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    subset='training'  # Usa os dados para treinamento
)

# Gerador de dados de validação
val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(48, 48),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation'  # Usa os dados para validação
)

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

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Parâmetros do modelo
input_shape = (48, 48, 1)
num_classes = 4
learning_rate = 1e-3

# Criação do modelo
model = build_vggnet(input_shape, num_classes)

# Compilação do modelo
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss=categorical_crossentropy,
              metrics=['accuracy', precision_m, recall_m, f1_m])

weights_path = 'saved_models/vggnet_emotions.h5'

# Tente carregar os pesos do modelo se eles existirem
try:
    model.load_weights(weights_path)
    print("Pesos carregados com sucesso.")
except Exception as e:
    print("Erro ao carregar os pesos:", str(e))

print("\n\n\n\n\n")
print(model.summary())
print("\n\n\n\n\n")

# Treinamento do modelo
epochs = 5
history = model.fit(train_generator,
                    epochs=epochs,
                    validation_data=val_generator,
                    )

# Salvando os pesos do modelo
model.save_weights(weights_path)

# Salvando os índices das classes
class_indices = train_generator.class_indices
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)
