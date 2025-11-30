# ============================================
# 🧩 BREAST CANCER DETECTOR - IA BINÁRIA (0/1)
# ============================================

# 0️⃣ Importações
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix

# 1️⃣ Configurações iniciais
base_path = '/kaggle/input/breast-cancer-dataset/Banco/Banco'   # caminho das pastas
planilha_path = '/kaggle/input/breast-cancer-dataset/PlanilhaCancer.xlsx'  # caminho da planilha

IMG_SIZE = (64, 164)
BATCH_SIZE = 32
EPOCHS = 10
AUTOTUNE = tf.data.experimental.AUTOTUNE

print("Pastas dentro de Banco:", os.listdir(base_path))

# 2️⃣ Ler a planilha e criar mapa de labels
df = pd.read_excel(planilha_path)
print("Colunas do Excel:", df.columns.tolist())
print(df.head())

df.columns = df.columns.str.strip()  # remove espaços
df['Imagem'] = df['Imagem'].astype(str)
df['Classificacao'] = df['Classificacao'].astype(int)

label_map = dict(zip(df['Imagem'], df['Classificacao']))
print(f"Total de imagens mapeadas: {len(label_map)}")
print("Exemplo de mapeamento:", list(label_map.items())[:5])

# 3️⃣ Detectar automaticamente se é grayscale (1 canal) ou colorida (3 canais)
def detect_image_channels(sample_folder, sample_limit=20):
    checked = 0
    channels = set()
    for root, _, files in os.walk(sample_folder):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, f)
                try:
                    im = Image.open(path)
                    if im.mode == 'L':
                        channels.add(1)
                    else:
                        channels.add(3)
                except:
                    pass
                checked += 1
                if checked >= sample_limit:
                    break
        if checked >= sample_limit:
            break
    return channels

channels = detect_image_channels(base_path)
N_CHANNELS = 1 if 1 in channels and 3 not in channels else 3
print("Canais detectados:", N_CHANNELS)

# 4️⃣ Listar imagens e labels nas pastas Treinar / Validar / Testar
def gather_files_and_labels(subfolder):
    folder = os.path.join(base_path, subfolder)
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files_sorted = sorted(files)
    paths = [os.path.join(folder, f) for f in files_sorted]
    labels = [label_map.get(f, None) for f in files_sorted]
    filtered = [(p, l) for p, l in zip(paths, labels) if l is not None]
    if len(filtered) != len(paths):
        print(f"Aviso: {len(paths) - len(filtered)} imagens sem rótulo em {subfolder}.")
    if not filtered:
        return [], []
    paths, labels = zip(*filtered)
    return list(paths), list(labels)

train_paths, train_labels = gather_files_and_labels('Treinar')
val_paths, val_labels = gather_files_and_labels('Validar')
test_paths, test_labels = gather_files_and_labels('Testar')

print("Treinar:", len(train_paths), "Validar:", len(val_paths), "Testar:", len(test_paths))

# 5️⃣ Criar dataset TensorFlow
def preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=N_CHANNELS, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMG_SIZE)
    return image, tf.cast(label, tf.float32)

def make_dataset(paths, labels, shuffle=True, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths))
    ds = ds.map(lambda p, l: tf.py_function(func=preprocess_image, inp=[p, l], Tout=(tf.float32, tf.float32)),
                num_parallel_calls=AUTOTUNE)
    def set_shape(img, lbl):
        img.set_shape([IMG_SIZE[0], IMG_SIZE[1], N_CHANNELS])
        lbl.set_shape([])
        return img, lbl
    ds = ds.map(set_shape, num_parallel_calls=AUTOTUNE)

    if augment:
        def augment_fn(img, lbl):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.08)
            return img, lbl
        ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

train_ds = make_dataset(train_paths, train_labels, shuffle=True, augment=True)
val_ds = make_dataset(val_paths, val_labels, shuffle=False)
test_ds = make_dataset(test_paths, test_labels, shuffle=False)

# 6️⃣ Construir modelo CNN
def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='AUC')])
    return model

input_shape = (IMG_SIZE[0], IMG_SIZE[1], N_CHANNELS)
model = build_model(input_shape)
model.summary()

# 7️⃣ Callbacks
checkpoint_path = "/kaggle/working/best_model.h5"
cb = [
    callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max'),
    callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)
]

# 8️⃣ Treinamento (50 épocas)
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=EPOCHS,
                    callbacks=cb)

# 9️⃣ Gráficos de perda e acurácia
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

# 🔟 Avaliar no conjunto de teste
model.load_weights(checkpoint_path)
loss, acc, auc = model.evaluate(test_ds)
print(f"\nDesempenho no Teste → Loss: {loss:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f}")

# 🔢 Previsões e relatório
y_true, y_pred_probs = [], []
for imgs, labels in test_ds:
    preds = model.predict(imgs)
    y_pred_probs.extend(preds.reshape(-1))
    y_true.extend(labels.numpy())

y_pred = [1 if p >= 0.5 else 0 for p in y_pred_probs]
print("\nRelatório de Classificação:")
print(classification_report(y_true, y_pred, digits=4))

cm = confusion_matrix(y_true, y_pred)
print("\nMatriz de Confusão:\n", cm)

plt.figure(figsize=(4,4))
plt.imshow(cm, cmap='Blues')
plt.title('Matriz de Confusão')
plt.xticks([0,1], ['Sem Câncer (0)', 'Com Câncer (1)'])
plt.yticks([0,1], ['Sem Câncer (0)', 'Com Câncer (1)'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha='center', va='center', color='black')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()

# 💾 Salvar resultados
out_df = pd.DataFrame({
    'Imagem': [os.path.basename(p) for p in test_paths],
    'Real': y_true,
    'Probabilidade(1)': y_pred_probs,
    'Previsto': y_pred
})
out_df.to_csv('/kaggle/working/resultados_teste.csv', index=False)
print("\nResultados salvos em: /kaggle/working/resultados_teste.csv ✅")

