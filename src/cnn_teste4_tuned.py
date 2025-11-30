# ============================================
# 🧩 BREAST CANCER DETECTOR  IA BINÁRIA (0/1)
# 🔍 400x400 + Dense 512 + Dropout 0.7 + LR 1e-5
# ============================================

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix

# ⚙️ Configurações
base_path = '/kaggle/input/breast-cancer-dataset/Banco/Banco'
planilha_path = '/kaggle/input/breast-cancer-dataset/PlanilhaCancer.xlsx'

IMG_SIZE = (400, 400)
BATCH_SIZE = 8           # menor batch por causa do tamanho grande
EPOCHS = 60
AUTOTUNE = tf.data.experimental.AUTOTUNE

# 📘 Ler planilha
df = pd.read_excel(planilha_path)
df.columns = df.columns.str.strip()
df['Imagem'] = df['Imagem'].astype(str)
df['Classificacao'] = df['Classificacao'].astype(int)
label_map = dict(zip(df['Imagem'], df['Classificacao']))

# 🔍 Detectar canais
def detect_channels(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(os.path.join(root, f))
                return 1 if img.mode == 'L' else 3
    return 3

N_CHANNELS = detect_channels(base_path)
print("Canais detectados:", N_CHANNELS)

# 📂 Função para listar imagens e labels
def gather_files(subfolder):
    folder = os.path.join(base_path, subfolder)
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    paths = [os.path.join(folder, f) for f in sorted(files)]
    labels = [label_map.get(f, None) for f in sorted(files)]
    filtered = [(p, l) for p, l in zip(paths, labels) if l is not None]
    if not filtered:
        return [], []
    paths, labels = zip(*filtered)
    return list(paths), list(labels)

train_paths, train_labels = gather_files('Treinar')
val_paths, val_labels = gather_files('Validar')
test_paths, test_labels = gather_files('Testar')

# 🧩 Preprocessamento
def preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=N_CHANNELS, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, IMG_SIZE)
    return img, tf.cast(label, tf.float32)

def make_dataset(paths, labels, augment=False, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths))
    ds = ds.map(lambda p, l: preprocess(p, l), num_parallel_calls=AUTOTUNE)
    if augment:
        def aug(img, lbl):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.05)
            img = tf.image.random_contrast(img, 0.9, 1.1)
            return img, lbl
        ds = ds.map(aug, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

train_ds = make_dataset(train_paths, train_labels, augment=True)
val_ds = make_dataset(val_paths, val_labels)
test_ds = make_dataset(test_paths, test_labels, shuffle=False)

# 🧠 Modelo CNN profundo e refinado
def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.7),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='AUC')])
    return model

input_shape = (IMG_SIZE[0], IMG_SIZE[1], N_CHANNELS)
model = build_model(input_shape)
model.summary()

# 🛠️ Callbacks
checkpoint_path = "/kaggle/working/best_model_v4.keras"
cb = [
    callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max'),
    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

# 🏋️ Treinamento
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb)

# 📊 Gráficos
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.legend(); plt.title('Perda')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.legend(); plt.title('Acurácia')
plt.show()

# 🧾 Avaliar
model.load_weights(checkpoint_path)
loss, acc, auc = model.evaluate(test_ds)
print(f"\nDesempenho no Teste → Loss: {loss:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f}")

# 🔢 Relatório
y_true, y_pred_probs = [], []
for imgs, labels in test_ds:
    preds = model.predict(imgs)
    y_pred_probs.extend(preds.reshape(-1))
    y_true.extend(labels.numpy())

y_pred = [1 if p >= 0.5 else 0 for p in y_pred_probs]
print("\nRelatório de Classificação:")
print(classification_report(y_true, y_pred, digits=4))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(4,4))
plt.imshow(cm, cmap='Blues')
plt.title('Matriz de Confusão')
plt.xticks([0,1], ['Sem Câncer', 'Com Câncer'])
plt.yticks([0,1], ['Sem Câncer', 'Com Câncer'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha='center', va='center', color='black')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()
