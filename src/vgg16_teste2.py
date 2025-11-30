# =========================================================================
# 🎯 CÓDIGO: VGG16, FOCAL LOSS, E IMAGENS PRÉ-PROCESSADAS (HEURISTICA)
# =========================================================================

# 0️⃣ Importações
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import VGG16
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

# 1️⃣ Configurações iniciais
# 🚨 CAMINHOS AJUSTADOS: Usando 'banco510imagens' e apontando a base de imagens para o Banco
BASE_DATASET_ROOT = '/kaggle/input/banco510imagens/BancoDados/Banco' 

# AJUSTE CRÍTICO: Base path agora aponta diretamente para o Banco,
# assumindo que as imagens da Heuristica estão dentro de Treinar/Validar/Testar
base_path = BASE_DATASET_ROOT  
planilha_path = os.path.join(BASE_DATASET_ROOT, 'Planilha.xlsx') 

IMG_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 30 
EPOCHS_PHASE_1 = 10 
AUTOTUNE = tf.data.experimental.AUTOTUNE
N_CHANNELS = 3

# ---------------------------------------------
# IMPLEMENTAÇÃO FOCAL LOSS
# ---------------------------------------------
def categorical_focal_loss(gamma=2.0, alpha=0.5):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        alpha_factor = (y_true * alpha) + ((1 - y_true) * (1 - alpha))
        focal_loss = modulating_factor * alpha_factor * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
    return focal_loss_fixed

FOCAL_LOSS = categorical_focal_loss(gamma=2.0, alpha=0.5)

# 2️⃣ Ler planilha e mapear labels
df = pd.read_excel(planilha_path)
df.columns = df.columns.str.strip()
df['Imagem'] = df['Imagem'].astype(str)
df['Classificacao'] = df['Classificacao'].astype(int)
label_map = dict(zip(df['Imagem'], df['Classificacao']))

# 3️⃣ Listar imagens e labels
def gather_files_and_labels(subfolder):
    # Procura em /kaggle/input/banco510imagens/BancoDados/Banco/{subfolder}
    folder = os.path.join(base_path, subfolder) 
    
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Pasta não encontrada: {folder}. Verifique se as pastas Treinar, Validar e Testar estão no nível correto.")
        
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files_sorted = sorted(files)
    paths = [os.path.join(folder, f) for f in files_sorted]
    labels = [label_map.get(f, None) for f in files_sorted] 
    filtered = [(p, l) for p, l in zip(paths, labels) if l is not None]
    if not filtered: 
        print(f"⚠️ Aviso: Nenhuma imagem encontrada ou mapeada na pasta {folder}.")
        return [], []
    paths, labels = zip(*filtered)
    return list(paths), list(labels)

train_paths, train_labels = gather_files_and_labels('Treinar')
val_paths, val_labels = gather_files_and_labels('Validar')
test_paths, test_labels = gather_files_and_labels('Testar')

# 4️⃣ Calcular pesos de classe
train_labels_np = np.array(train_labels)
class_weights = class_weight.compute_class_weight('balanced', 
                                                  classes=np.unique(train_labels_np), 
                                                  y=train_labels_np)
class_weight_dict = dict(enumerate(class_weights))
print("\n⚖️ Pesos de Classe (Class Weight) para balanceamento:", class_weight_dict)

# 5️⃣ Criar datasets (SIMPLIFICADO - Imagens já pré-processadas)
zoom_layer = tf.keras.layers.RandomZoom(height_factor=(-0.2, 0.0), width_factor=(-0.2, 0.0), fill_mode='nearest')
translation_layer = tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='nearest')

def preprocess_image(path, label):
    # Imagem já cortada! Apenas redimensionamos para 512x512
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=N_CHANNELS, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMG_SIZE)
    return image, tf.cast(label, tf.float32)

def make_dataset(paths, labels, shuffle=True, augment=False, zoom_op=None, translate_op=None):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle: ds = ds.shuffle(buffer_size=len(paths))
    
    ds = ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
                
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


# 6️⃣ Modelo (VGG16)
def build_vgg_model(input_shape, initial_learning_rate):
    base_model = VGG16(weights='imagenet', 
                          include_top=False,
                          input_shape=input_shape)
    base_model.trainable = False 
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'), 
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid') 
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate), 
                  loss=FOCAL_LOSS, 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='AUC')])
    
    return base_model, model

input_shape = (IMG_SIZE[0], IMG_SIZE[1], N_CHANNELS)
BASE_MODEL, MODEL = build_vgg_model(input_shape, initial_learning_rate=1e-4)
MODEL.summary()

# 7️⃣ Callbacks
checkpoint_path = "/kaggle/working/best_model.keras"
cb = [
    callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max'),
    callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-8, verbose=1)
]


# 8️⃣ Treinamento em Duas Fases (Fine-Tuning)
print(f"\n🚀 FASE 1: Treinando a Cabeça (Head) do VGG16 com Focal Loss.")
print("=============================================================================================================")
history_phase1 = MODEL.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_PHASE_1, class_weight=class_weight_dict) 

# --- FASE 2: FINE-TUNING ---
print("\n\n🚀 FASE 2: Iniciando Fine-Tuning. Descongelando Base do VGG16 e usando LR muito baixo.")
print("==========================================================================")
BASE_MODEL.trainable = True
for layer in BASE_MODEL.layers[:4]: # Congelando as primeiras 4 camadas do VGG
    layer.trainable = False

fine_tune_lr = 1e-7 
MODEL.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr), 
              loss=FOCAL_LOSS, 
              metrics=['accuracy', tf.keras.metrics.AUC(name='AUC')])

total_epochs = EPOCHS_PHASE_1 + (EPOCHS - EPOCHS_PHASE_1) 
history_phase2 = MODEL.fit(train_ds,
                           validation_data=val_ds,
                           epochs=total_epochs, 
                           initial_epoch=history_phase1.epoch[-1] + 1, 
                           callbacks=cb,
                           class_weight=class_weight_dict)

# 9️⃣ Gráficos
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



3° Teste:
# =========================================================================
# 🎯 CÓDIGO FINAL AJUSTADO: VGG16, FOCAL LOSS (ALPHA=0.25), E IMAGENS PRÉ-PROCESSADAS
# =========================================================================

# 0️⃣ Importações
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import VGG16
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

# 1️⃣ Configurações iniciais
# 🚨 CAMINHOS AJUSTADOS: Usando 'banco510imagens' e apontando a base de imagens para o Banco
BASE_DATASET_ROOT = '/kaggle/input/banco510imagens/BancoDados/Banco' 

# AJUSTE CRÍTICO: Base path agora aponta diretamente para o Banco,
# assumindo que as imagens da Heuristica estão dentro de Treinar/Validar/Testar
base_path = BASE_DATASET_ROOT  
planilha_path = os.path.join(BASE_DATASET_ROOT, 'Planilha.xlsx') 

IMG_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 30 
EPOCHS_PHASE_1 = 10 
AUTOTUNE = tf.data.experimental.AUTOTUNE
N_CHANNELS = 3

# ---------------------------------------------
# IMPLEMENTAÇÃO FOCAL LOSS (ALPHA AJUSTADO)
# ---------------------------------------------
# NOVO: Constantes para gerenciar e exibir os hiperparâmetros
FOCAL_LOSS_ALPHA = 0.25  # Ajustado para 0.25 para penalizar mais erros na Classe 0 (Falsos Positivos)
FOCAL_LOSS_GAMMA = 2.0

def categorical_focal_loss(gamma=FOCAL_LOSS_GAMMA, alpha=FOCAL_LOSS_ALPHA): 
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        alpha_factor = (y_true * alpha) + ((1 - y_true) * (1 - alpha))
        focal_loss = modulating_factor * alpha_factor * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
    return focal_loss_fixed

FOCAL_LOSS = categorical_focal_loss()

# 2️⃣ Ler planilha e mapear labels
df = pd.read_excel(planilha_path)
df.columns = df.columns.str.strip()
df['Imagem'] = df['Imagem'].astype(str)
df['Classificacao'] = df['Classificacao'].astype(int)
label_map = dict(zip(df['Imagem'], df['Classificacao']))

# 3️⃣ Listar imagens e labels
def gather_files_and_labels(subfolder):
    # Procura em /kaggle/input/banco510imagens/BancoDados/Banco/{subfolder}
    folder = os.path.join(base_path, subfolder) 
    
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Pasta não encontrada: {folder}. Verifique se as pastas Treinar, Validar e Testar estão no nível correto.")
        
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files_sorted = sorted(files)
    paths = [os.path.join(folder, f) for f in files_sorted]
    labels = [label_map.get(f, None) for f in files_sorted] 
    filtered = [(p, l) for p, l in zip(paths, labels) if l is not None]
    if not filtered: 
        print(f"⚠️ Aviso: Nenhuma imagem encontrada ou mapeada na pasta {folder}.")
        return [], []
    paths, labels = zip(*filtered)
    return list(paths), list(labels)

train_paths, train_labels = gather_files_and_labels('Treinar')
val_paths, val_labels = gather_files_and_labels('Validar')
test_paths, test_labels = gather_files_and_labels('Testar')

# 4️⃣ Calcular pesos de classe
train_labels_np = np.array(train_labels)
class_weights = class_weight.compute_class_weight('balanced', 
                                                 classes=np.unique(train_labels_np), 
                                                 y=train_labels_np)
class_weight_dict = dict(enumerate(class_weights))
print("\n⚖️ Pesos de Classe (Class Weight) para balanceamento:", class_weight_dict)

# 5️⃣ Criar datasets (SIMPLIFICADO - Imagens já pré-processadas)
zoom_layer = tf.keras.layers.RandomZoom(height_factor=(-0.2, 0.0), width_factor=(-0.2, 0.0), fill_mode='nearest')
translation_layer = tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='nearest')

def preprocess_image(path, label):
    # Imagem já cortada! Apenas redimensionamos para 512x512
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=N_CHANNELS, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMG_SIZE)
    return image, tf.cast(label, tf.float32)

def make_dataset(paths, labels, shuffle=True, augment=False, zoom_op=None, translate_op=None):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle: ds = ds.shuffle(buffer_size=len(paths))
    
    ds = ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
                
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


# 6️⃣ Modelo (VGG16)
def build_vgg_model(input_shape, initial_learning_rate):
    base_model = VGG16(weights='imagenet', 
                              include_top=False,
                              input_shape=input_shape)
    base_model.trainable = False 
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'), 
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid') 
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate), 
                  loss=FOCAL_LOSS, 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='AUC')])
    
    return base_model, model

input_shape = (IMG_SIZE[0], IMG_SIZE[1], N_CHANNELS)
BASE_MODEL, MODEL = build_vgg_model(input_shape, initial_learning_rate=1e-4)
MODEL.summary()

# 7️⃣ Callbacks
checkpoint_path = "/kaggle/working/best_model.keras"
cb = [
    callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max'),
    callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-8, verbose=1)
]


# 8️⃣ Treinamento em Duas Fases (Fine-Tuning)
# CORREÇÃO APLICADA AQUI: Uso da variável FOCAL_LOSS_ALPHA
print(f"\n🚀 FASE 1: Treinando a Cabeça (Head) do VGG16 com Focal Loss (Alpha={FOCAL_LOSS_ALPHA}).")
print("=============================================================================================================")
history_phase1 = MODEL.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_PHASE_1, class_weight=class_weight_dict) 

# --- FASE 2: FINE-TUNING ---
print("\n\n🚀 FASE 2: Iniciando Fine-Tuning. Descongelando Base do VGG16 e usando LR muito baixo.")
print("==========================================================================")
BASE_MODEL.trainable = True
for layer in BASE_MODEL.layers[:4]: # Congelando as primeiras 4 camadas do VGG
    layer.trainable = False

fine_tune_lr = 1e-7 
MODEL.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr), 
              loss=FOCAL_LOSS, 
              metrics=['accuracy', tf.keras.metrics.AUC(name='AUC')])

total_epochs = EPOCHS_PHASE_1 + (EPOCHS - EPOCHS_PHASE_1) 
history_phase2 = MODEL.fit(train_ds,
                           validation_data=val_ds,
                           epochs=total_epochs, 
                           initial_epoch=history_phase1.epoch[-1] + 1, 
                           callbacks=cb,
                           class_weight=class_weight_dict)

# 9️⃣ Gráficos
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

# THRESHOLD MANTIDO em 0.45, mas pode ser ajustado para 0.55 ou 0.60 se o viés persistir.
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

