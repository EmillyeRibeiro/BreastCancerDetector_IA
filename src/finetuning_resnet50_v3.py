# ============================================
# 🧩 BREAST CANCER DETECTOR - IA BINÁRIA (0/1)
# ============================================

# 0️⃣ Importações
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

# 1️⃣ Configurações iniciais
# **Certifique-se de que estas variáveis de caminho estejam definidas**
# base_path = '/kaggle/input/bancocom510imagens/BancoDados/Banco' 
# planilha_path = '/kaggle/input/bancocom510imagens/BancoDados/Banco/Planilha.xlsx' 

IMG_SIZE = (512, 512)
BATCH_SIZE = 4 
EPOCHS = 10
AUTOTUNE = tf.data.experimental.AUTOTUNE
N_CHANNELS = 1 

# 2️⃣ Ler planilha e mapear labels
df = pd.read_excel(planilha_path)
df.columns = df.columns.str.strip()
df['Imagem'] = df['Imagem'].astype(str)
df['Classificacao'] = df['Classificacao'].astype(int)

label_map = dict(zip(df['Imagem'], df['Classificacao']))
print(f"Total de imagens mapeadas: {len(label_map)}")

# 3️⃣ Listar imagens e labels
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

# 4️⃣ Calcular pesos de classe para Treinamento
train_labels_np = np.array(train_labels)
class_weights = class_weight.compute_class_weight('balanced', 
                                                  classes=np.unique(train_labels_np), 
                                                  y=train_labels_np)

class_weight_dict = dict(enumerate(class_weights))
print("\n⚖️ Pesos de Classe (Class Weight) para balanceamento:", class_weight_dict)


# 5️⃣ Criar datasets (CORREÇÃO DE RandomZoom/RandomTranslation)

# ➡️ CORREÇÃO: As camadas Keras que criam variáveis devem ser INSTANCIADAS UMA VEZ
zoom_layer = tf.keras.layers.RandomZoom(height_factor=(-0.2, 0.0), width_factor=(-0.2, 0.0), fill_mode='nearest')
translation_layer = tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='nearest')

def preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=N_CHANNELS, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMG_SIZE)
    return image, tf.cast(label, tf.float32)

# Modificamos make_dataset para aceitar as instâncias de camadas
def make_dataset(paths, labels, shuffle=True, augment=False, zoom_op=None, translate_op=None):
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
        # A função augment_fn agora usa as instâncias passadas (zoom_op, translate_op)
        def augment_fn(img, lbl):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, 0.1)
            img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
            img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)) 
            
            # ⬅️ APLICA AS INSTÂNCIAS (NÃO AS CRIA NOVAMENTE)
            if zoom_op:
                img = zoom_op(img)
            if translate_op:
                img = translate_op(img)
                
            return img, lbl
        ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

# Passando as instâncias para a criação do dataset de treino
train_ds = make_dataset(train_paths, train_labels, augment=True, zoom_op=zoom_layer, translate_op=translation_layer)
val_ds = make_dataset(val_paths, val_labels, augment=False)
test_ds = make_dataset(test_paths, test_labels, augment=False)


# 6️⃣ Modelo CNN (Simplificado: 6 Bloques Convolucionais + Apenas 1 Camada Densa de Saída)
def build_model_simplified(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Blocos Convolucionais Profundos
        layers.Conv2D(32, 3, activation='relu', padding='same'), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same'), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu', padding='same'), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(256, 3, activation='relu', padding='same'), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(512, 3, activation='relu', padding='same'), layers.BatchNormalization(), layers.MaxPooling2D(),
        
        # Camadas densas (Apenas a camada de saída)
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid') 
    ])
    
    # Otimizador Nadam com LR 5e-5
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=5e-5), 
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='AUC')])
    return model

input_shape = (IMG_SIZE[0], IMG_SIZE[1], N_CHANNELS)
model = build_model_simplified(input_shape)
model.summary()


# 7️⃣ Callbacks
checkpoint_path = "/kaggle/working/best_model.keras"
cb = [
    callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max'),
    callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1) 
]


# 8️⃣ Treinamento
print(f"\n🚀 Iniciando Treinamento com IMG_SIZE={IMG_SIZE} e BATCH_SIZE={BATCH_SIZE}.")
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=EPOCHS,
                    callbacks=cb,
                    class_weight=class_weight_dict) 

# 9️⃣ Gráficos
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Treino'); plt.plot(history.history['val_loss'], label='Validação')
plt.legend(); plt.title('Perda')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Treino'); plt.plot(history.history['val_accuracy'], label='Validação')
plt.legend(); plt.title('Acurácia')
plt.show()

# 🔟 Avaliar no teste
model.load_weights(checkpoint_path)
loss, acc, auc = model.evaluate(test_ds)
print(f"\nDesempenho no Teste FINAL → Loss: {loss:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f}")

# 🔢 Previsões e relatório
y_true, y_pred_probs = [], []
for imgs, labels in test_ds:
    preds = model.predict(imgs, verbose=0)
    y_pred_probs.extend(preds.reshape(-1))
    y_true.extend(labels.numpy())

# Limiar de 0.45 para aumentar a Sensibilidade
THRESHOLD = 0.45
y_pred = [1 if p >= THRESHOLD else 0 for p in y_pred_probs] 
print(f"\nRelatório de Classificação (Threshold = {THRESHOLD}):")
print(classification_report(y_true, y_pred, digits=4))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(4,4))
plt.imshow(cm, cmap='Blues')
plt.title(f'Matriz de Confusão (T={THRESHOLD})')
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

# 💾 Salvar modelo
model.save("/kaggle/working/final_model.keras")
print("\n✅ Modelo salvo no formato moderno (.keras) com sucesso!")

