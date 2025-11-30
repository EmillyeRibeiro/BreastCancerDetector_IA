# ============================================
# 🧩 BREAST CANCER DETECTOR - IA BINÁRIA (FINE-TUNING 512x512)
# ============================================

# 0️⃣ Importações (Mantidas)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# As importações do TF devem vir após a definição da Focal Loss
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight


# 1️⃣ Configurações iniciais
base_path = '../input/bancocom510imagens/BancoDados/Banco'
planilha_path = '../input/bancocom510imagens/BancoDados/Banco/Planilha.xlsx'

# 🚀 Resolução Aumentada e Batch Size Reduzido
IMG_SIZE = (512, 512)
BATCH_SIZE = 8
EPOCHS = 50 
EPOCHS_PHASE_1 = 10 
AUTOTUNE = tf.data.experimental.AUTOTUNE
N_CHANNELS = 3

# 2️⃣ Ler planilha e mapear labels (Mantidas)
df = pd.read_excel(planilha_path)
df.columns = df.columns.str.strip()
df['Imagem'] = df['Imagem'].astype(str)
df['Classificacao'] = df['Classificacao'].astype(int)
label_map = dict(zip(df['Imagem'], df['Classificacao']))

# 3️⃣ Listar imagens e labels (Mantidas)
def gather_files_and_labels(subfolder):
    folder = os.path.join(base_path, subfolder)
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files_sorted = sorted(files)
    paths = [os.path.join(folder, f) for f in files_sorted]
    labels = [label_map.get(f, None) for f in files_sorted]
    filtered = [(p, l) for p, l in zip(paths, labels) if l is not None]
    if not filtered: return [], []
    paths, labels = zip(*filtered)
    return list(paths), list(labels)

train_paths, train_labels = gather_files_and_labels('Treinar')
val_paths, val_labels = gather_files_and_labels('Validar')
test_paths, test_labels = gather_files_and_labels('Testar')

# 4️⃣ Calcular pesos de classe para Treinamento (Mantidos)
train_labels_np = np.array(train_labels)
class_weights = class_weight.compute_class_weight('balanced', 
                                                  classes=np.unique(train_labels_np), 
                                                  y=train_labels_np)
class_weight_dict = dict(enumerate(class_weights))
print("\n⚖️ Pesos de Classe (Class Weight) para balanceamento:", class_weight_dict)
# NOTA: O class_weight ainda é útil para Focal Loss, mas a perda tem seu próprio termo alpha (0.5 neste caso).

# 5️⃣ Criar datasets (Mantidas)
zoom_layer = tf.keras.layers.RandomZoom(height_factor=(-0.2, 0.0), width_factor=(-0.2, 0.0), fill_mode='nearest')
translation_layer = tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='nearest')

def preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=N_CHANNELS, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMG_SIZE)
    return image, tf.cast(label, tf.float32)

def make_dataset(paths, labels, shuffle=True, augment=False, zoom_op=None, translate_op=None):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle: ds = ds.shuffle(buffer_size=len(paths))
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
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, 0.1)
            img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
            img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)) 
            if zoom_op: img = zoom_op(img)
            if translate_op: img = translate_op(img)
            return img, lbl
        ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

train_ds = make_dataset(train_paths, train_labels, augment=True, zoom_op=zoom_layer, translate_op=translation_layer)
val_ds = make_dataset(val_paths, val_labels, augment=False)
test_ds = make_dataset(test_paths, test_labels, augment=False)


# 6️⃣ Modelo (ResNet50 com Função de Retorno Base)
def build_resnet_model(input_shape, initial_learning_rate):
    base_model = ResNet50(weights='imagenet', 
                          include_top=False,
                          input_shape=input_shape)
    base_model.trainable = False 
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(), 
        layers.Dense(128, activation='relu'), 
        layers.Dropout(0.2), # ⬅️ DROPOUT REDUZIDO
        layers.Dense(1, activation='sigmoid') 
    ])
    
    # Compilação Fase 1: Usando Adam e Focal Loss
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate), 
                  loss=FOCAL_LOSS, # ⬅️ USANDO FOCAL LOSS
                  metrics=['accuracy', tf.keras.metrics.AUC(name='AUC')])
    
    return base_model, model

input_shape = (IMG_SIZE[0], IMG_SIZE[1], N_CHANNELS)
BASE_MODEL, MODEL = build_resnet_model(input_shape, initial_learning_rate=1e-4)
MODEL.summary()

# 7️⃣ Callbacks (Mantidos)
checkpoint_path = "/kaggle/working/best_model.keras"
cb = [
    callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max'),
    callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-8, verbose=1)
]


# 8️⃣ Treinamento em Duas Fases (Fine-Tuning)
print(f"\n🚀 FASE 1: Treinando a Cabeça (Head) do Modelo com FOCAL LOSS. (IMG_SIZE={IMG_SIZE}, BATCH_SIZE={BATCH_SIZE})")
print("=============================================================================================================")
history_phase1 = MODEL.fit(train_ds,
                           validation_data=val_ds,
                           epochs=EPOCHS_PHASE_1,
                           class_weight=class_weight_dict) 

# --- FASE 2: FINE-TUNING ---
print("\n\n🚀 FASE 2: Iniciando Fine-Tuning. Descongelando Base e usando LR muito baixo.")
print("==========================================================================")
BASE_MODEL.trainable = True
# ⬅️ Congelando menos camadas para melhor adaptação
for layer in BASE_MODEL.layers[:100]: # 100 em vez de 140
    layer.trainable = False

# Recompilar com Taxa de Aprendizado MUITO BAIXA (crítico para Fine-Tuning)
fine_tune_lr = 1e-7 
MODEL.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr), 
              loss=FOCAL_LOSS, # ⬅️ USANDO FOCAL LOSS
              metrics=['accuracy', tf.keras.metrics.AUC(name='AUC')])

total_epochs = EPOCHS_PHASE_1 + (EPOCHS - EPOCHS_PHASE_1) 
history_phase2 = MODEL.fit(train_ds,
                           validation_data=val_ds,
                           epochs=total_epochs, 
                           initial_epoch=history_phase1.epoch[-1] + 1, 
                           callbacks=cb,
                           class_weight=class_weight_dict)

# 9️⃣ Gráficos (Combinando históricos)
hist_loss = history_phase1.history['loss'] + history_phase2.history['loss']
hist_val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']
hist_acc = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
hist_val_acc = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(hist_loss, label='Treino'); plt.plot(hist_val_loss, label='Validação')
plt.axvline(x=EPOCHS_PHASE_1 - 1, color='r', linestyle='--', label='Início Fine-Tuning')
plt.legend(); plt.title('Perda')

plt.subplot(1,2,2)
plt.plot(hist_acc, label='Treino'); plt.plot(hist_val_acc, label='Validação')
plt.axvline(x=EPOCHS_PHASE_1 - 1, color='r', linestyle='--', label='Início Fine-Tuning')
plt.legend(); plt.title('Acurácia')
plt.show()

# 🔟 Avaliar no teste e Previsões
MODEL.load_weights(checkpoint_path)
loss, acc, auc = MODEL.evaluate(test_ds)
print(f"\nDesempenho no Teste FINAL → Loss: {loss:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f}")

y_true, y_pred_probs = [], []
for imgs, labels in test_ds:
    preds = MODEL.predict(imgs, verbose=0)
    y_pred_probs.extend(preds.reshape(-1))
    y_true.extend(labels.numpy())

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

# 💾 Salvar resultados e modelo
out_df = pd.DataFrame({
    'Imagem': [os.path.basename(p) for p in test_paths],
    'Real': y_true,
    'Probabilidade(1)': y_pred_probs,
    'Previsto': y_pred
})
out_df.to_csv('/kaggle/working/resultados_teste.csv', index=False)
print("\nResultados salvos em: /kaggle/working/resultados_teste.csv ✅")
MODEL.save("/kaggle/working/final_model.keras")
print("\n✅ Modelo salvo no formato moderno (.keras) com sucesso!")

