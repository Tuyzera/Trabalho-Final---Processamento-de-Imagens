import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras import layers, models, callbacks
from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import re

# ================= CONFIGURAÇÕES =================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
IMGS_PER_LOCAL = 15
DATA_DIR = "dataset"
EXTRA_PREDICTION_DIR = "img_prediction" # Pasta com as imagens de 2000-2020
OUTPUT_DIR = "resultado_final_oficial"
CLASS_THRESHOLD = 2.0  # Se tiver mais de 2% de branco, é DESMATADO

# Cria as pastas
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "relatorios_visuais"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "graficos_longo_prazo"), exist_ok=True)

tf.random.set_seed(42); np.random.seed(42)

# ================= 1. PRÉ-PROCESSAMENTO =================
def process_image_mask(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)

    # 1. Solo Pálido / Branco (Areia, Terra seca)
    lower_pale = np.array([0, 0, 100]) 
    upper_pale = np.array([45, 90, 255])
    mask_pale = cv2.inRange(hsv, lower_pale, upper_pale)

    # 2. Solo Avermelhado / Marrom (Terra úmida/escura)
    lower_dark = np.array([0, 10, 30]) 
    upper_dark = np.array([30, 255, 200]) # Brilho aumentado para pegar terra clara
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

    # 3. Verde (Floresta) - Trava de segurança
    lower_green = np.array([35, 35, 30]) 
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # União
    mask_brown = cv2.bitwise_or(mask_pale, mask_dark)
    
    binary = np.zeros(IMG_SIZE, dtype=np.uint8)
    binary[mask_brown > 0] = 255
    
    # Limpeza: Remove falsos positivos onde é certeza que é floresta
    binary[mask_green > 0] = 0 

    # Fechamento morfológico para preencher buracos na terra
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    pct = (np.sum(binary == 255) / binary.size) * 100
    
    # Definição da Classe para o relatório
    label = 1 if pct > CLASS_THRESHOLD else 0
    
    return img_resized, binary, pct, label

# ================= 2. CARREGAMENTO =================
def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def get_metadata(folder_name):
    full_path = os.path.join(DATA_DIR, folder_name)
    files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        files.extend([str(p) for p in Path(full_path).rglob(ext)])
    files.sort(key=natural_sort_key)
    
    data = []
    for i, fpath in enumerate(files):
        local_idx = (i // IMGS_PER_LOCAL) + 1
        pos = i % IMGS_PER_LOCAL
        timestamp = float(f"{2015 + (pos//2)}.{1 if pos%2==0 else 2}")
        data.append({"path": fpath, "local": local_idx, "timestamp": timestamp})
    return pd.DataFrame(data)

def data_generator(df, batch_size=32, training=True):
    while True:
        if training: df = df.sample(frac=1)
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            X, y = [], []
            for _, row in batch.iterrows():
                try:
                    img = cv2.imread(row['path'])
                    if img is None: continue
                    rgb, _, pct, _ = process_image_mask(img)
                    X.append(rgb)
                    y.append(pct)
                except: continue
            if X: yield np.array(X), np.array(y)

# ================= 3. MODELO VENCEDOR =================
def build_cnn():
    inp = layers.Input(shape=(*IMG_SIZE, 3))
    x = layers.RandomRotation(0.15)(inp)
    x = layers.Rescaling(1./255)(x)
    
    for filters in [32, 64, 128]:
        x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.BatchNormalization()(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='linear')(x)
    
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ================= 4. RELATÓRIOS E PLOTS =================
def generate_full_report(model, test_df):
    print("\n>>> Gerando Relatório Completo (Classificação + Regressão)...")
    y_true_cls = []; y_true_reg = []; X_test = []
    
    for _, row in test_df.iterrows():
        img = cv2.imread(row['path'])
        if img is None: continue
        rgb, _, pct, label = process_image_mask(img)
        X_test.append(rgb); y_true_reg.append(pct); y_true_cls.append(label)
        
    X_test = np.array(X_test)
    preds_pct = model.predict(X_test).flatten()
    y_pred_cls = (preds_pct > CLASS_THRESHOLD).astype(int)
    # CLASS_THRESHOLD = 2.0

    print("\nRELATÓRIO DE CLASSIFICAÇÃO:")
    print(classification_report(y_true_cls, y_pred_cls, target_names=['Preservado', 'Desmatado'], zero_division=0))
    
    cm = confusion_matrix(y_true_cls, y_pred_cls)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Preservado', 'Desmatado'], yticklabels=['Preservado', 'Desmatado'])
    plt.title('Matriz de Confusão'); plt.xlabel('Previsto'); plt.ylabel('Real')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "matriz_confusao.png")); plt.close()
    
    mae = np.mean(np.abs(np.array(y_true_reg) - preds_pct))
    print(f"Erro Médio Absoluto (MAE) na Regressão: {mae:.2f}%")

def forecast_2025(df_val):
    results = []
    pcts = []
    for p in df_val['path']:
        _, _, pct, _ = process_image_mask(cv2.imread(p))
        pcts.append(pct)
    df_val['pct_real'] = pcts
    
    for loc in df_val['local'].unique():
        subset = df_val[df_val['local'] == loc].sort_values('timestamp')
        if len(subset) < 2: continue
        reg = Ridge(alpha=1.0)
        reg.fit(subset[['timestamp']].values, subset['pct_real'].values)
        pred = np.clip(reg.predict([[2025.1]])[0], 0, 100)
        results.append({"Local": loc, "Atual": subset['pct_real'].values[-1], "Previsao_2025": pred})
    return pd.DataFrame(results)

def plot_comparative_bars(df_res):
    labels = df_res['Local'].astype(str)
    x = np.arange(len(labels)); width = 0.35
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(x - width/2, df_res['Atual'], width, label='Atual', color='#2E86C1')
    ax.bar(x + width/2, df_res['Previsao_2025'], width, label='Previsão 2025.1', color='#C0392B')
    ax.set_ylabel('% Área Desmatada'); ax.set_title('Atual vs 2025')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45); ax.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "grafico_comparativo_2025.png")); plt.close()

def generate_visual_reports(df):
    plot_dir = os.path.join(OUTPUT_DIR, "relatorios_visuais")
    for loc in df['local'].unique():
        subset = df[df['local'] == loc].sort_values('timestamp')
        if len(subset) < 3: continue
        idxs = [0, len(subset)//2, len(subset)-1]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for col, idx in enumerate(idxs):
            row = subset.iloc[idx]
            img = cv2.imread(row['path'])
            rgb, binary, pct, _ = process_image_mask(img)
            axes[0, col].imshow(rgb); axes[0, col].set_title(f"{row['timestamp']}")
            axes[0, col].axis('off')
            axes[1, col].imshow(binary, cmap='gray'); axes[1, col].set_title(f"{pct:.1f}%")
            axes[1, col].axis('off')
        plt.suptitle(f"Local {loc}"); plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"Local_{loc}.png")); plt.close()

# ================= 5. NOVA FUNÇÃO: EVOLUÇÃO LONGO PRAZO (2000-2020) =================
def generate_long_term_graphs(model):
    if not os.path.exists(EXTRA_PREDICTION_DIR):
        print(f"\nPasta '{EXTRA_PREDICTION_DIR}' não encontrada. Pulando gráficos de 2000-2020.")
        return

    print(f"\n>>> Gerando Gráficos de Evolução Histórica (2000-2020)...")
    output_dir = os.path.join(OUTPUT_DIR, "graficos_longo_prazo")
    
    locais = [f for f in os.listdir(EXTRA_PREDICTION_DIR) if os.path.isdir(os.path.join(EXTRA_PREDICTION_DIR, f))]
    
    for local in locais:
        print(f"   Processando {local}...")
        path_local = os.path.join(EXTRA_PREDICTION_DIR, local)
        files = sorted(os.listdir(path_local))
        data = []
        
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                match = re.search(r'(\d{4})', f)
                if match:
                    year = int(match.group(1))
                    img_path = os.path.join(path_local, f)
                    try:
                        img = cv2.imread(img_path)
                        if img is None: continue
                        rgb, _, _, _ = process_image_mask(img)
                        img_batch = np.expand_dims(rgb, axis=0)
                        pred_pct = model.predict(img_batch, verbose=0)[0][0]
                        data.append({'Year': year, 'Pct': pred_pct})
                    except: continue
        
        if not data: continue
        
        df = pd.DataFrame(data).sort_values('Year')
        
        reg = Ridge(alpha=1.0)
        reg.fit(df[['Year']].values, df['Pct'].values)
        pred_2025 = float(np.clip(reg.predict([[2025]]), 0, 100)[0])
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(df['Year'], df['Pct'], marker='o', linewidth=2, label='Histórico (CNN)')
        plt.plot([df['Year'].iloc[-1], 2025], [df['Pct'].iloc[-1], pred_2025], 'r--', label='Projeção 2025')
        plt.scatter([2025], [pred_2025], color='red', s=100, zorder=5)
        plt.title(f'Evolução do Desmatamento (2000-2025) - {local}')
        plt.xlabel('Ano'); plt.ylabel('% Desmatamento')
        
        # --- AJUSTE DINÂMICO DO EIXO Y ---
        # Pega o maior valor (histórico ou previsão)
        max_val = max(df['Pct'].max(), pred_2025)
        # Define o limite como o máximo + 10% de margem (mas trava em 105% se passar)
        upper_limit = min(105, max_val + 10)
        plt.ylim(0, upper_limit)
        plt.grid(True, alpha=0.3); plt.legend()
        plt.savefig(os.path.join(output_dir, f"{local}_evolucao.png"))
        plt.close()

# ================= MAIN =================
def main():
    print("=== PIPELINE FINAL ===")
    train_df = get_metadata("train_data")
    val_df = get_metadata("val_data")
    test_df = get_metadata("test_data")
    
    if train_df.empty: return

    print("\n>>> Treinando CNN...")
    model = build_cnn()
    model.fit(
        data_generator(train_df, BATCH_SIZE),
        steps_per_epoch=len(train_df) // BATCH_SIZE,
        validation_data=data_generator(val_df, BATCH_SIZE),
        validation_steps=len(val_df) // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[callbacks.EarlyStopping(patience=7, restore_best_weights=True)],
        verbose=1
    )
    
    # 1. Métricas (Classificação + Regressão)
    generate_full_report(model, test_df)
    
    # 2. Gráficos do Dataset Original (2015-2022)
    df_fore = forecast_2025(val_df)
    plot_comparative_bars(df_fore)
    generate_visual_reports(val_df)
    df_fore.to_csv(os.path.join(OUTPUT_DIR, "previsoes_2025_dataset.csv"), index=False)
    
    # 3. Gráficos de Evolução Longo Prazo (2000-2025)
    # Procura na pasta "img_prediction" e gera os gráficos extras
    generate_long_term_graphs(model)
    
    print(f"\n=== SUCESSO! Resultados salvos em '{OUTPUT_DIR}' ===")

if __name__ == "__main__":
    main()