# hpc_gpu_final.py
import os
import time
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler as CPU_StandardScaler

# ==========================================
# 1. DATOS REALES DE TU EJECUCIÓN CPU (BASELINE)
# ==========================================
# Extraídos de tus logs anteriores (Mejores tiempos N-Cores)
TIEMPOS_CPU_BASELINE = {
    'Phones Accelerometer': 84.73,  # N=12
    'Phones Gyroscope': 89.19,      # N=16
    'Watch Accelerometer': 57.67,   # N=8
    'Watch Gyroscope': 60.13        # N=4/8
}

# Intentar importar librerias de GPU
try:
    import cupy as cp
    from cuml.svm import SVC as GPU_SVC
    print("Librerias RAPIDS (CuPy/cuML) cargadas correctamente.")
except ImportError as e:
    print(f"ERROR CRITICO DE LIBRERIAS: {e}")
    exit()

# ================= CONFIGURACION =================
SENSORES_A_PROBAR = {
    'Phones Accelerometer': '/home/angel/Documentos/CÓMPUTO PARALELO/PROYECTO/Phones_accelerometer.csv',
    'Phones Gyroscope': '/home/angel/Documentos/CÓMPUTO PARALELO/PROYECTO/Phones_gyroscope.csv',
    'Watch Accelerometer': '/home/angel/Documentos/CÓMPUTO PARALELO/PROYECTO/Watch_accelerometer.csv',
    'Watch Gyroscope': '/home/angel/Documentos/CÓMPUTO PARALELO/PROYECTO/Watch_gyroscope.csv'
}

TAMANO_VENTANA = 128
PASO = 64

# ================= GESTION DE MEMORIA =================
def limpiar_gpu():
    """Limpia cache de CuPy y fuerza el recolector de basura"""
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp.fft.config.get_plan_cache().clear()
    except:
        pass
    gc.collect()

# ================= CARGA DE DATOS =================
def cargar_datos_ventaneados(archivo):
    if not os.path.exists(archivo): return None, None, None, None
    print(f"Leyendo CSV: {os.path.basename(archivo)}...")
    
    df = pd.read_csv(archivo, on_bad_lines='skip', usecols=['x', 'y', 'z', 'gt', 'Device'])
    df.dropna(inplace=True)
    df = df[df['gt'] != 'null']
    
    le_act = {lbl: i for i, lbl in enumerate(df['gt'].unique())}
    le_dev = {dev: i for i, dev in enumerate(df['Device'].unique())}
    
    data_vals = df[['x', 'y', 'z']].values.astype(np.float32)
    labels = df['gt'].map(le_act).values.astype(np.int32)
    devices = df['Device'].map(le_dev).values.astype(np.int32)
    
    X_wins = sliding_window_view(data_vals, window_shape=(TAMANO_VENTANA, 3))[::PASO]
    X_wins = X_wins.squeeze()
    
    idx = np.arange(TAMANO_VENTANA//2, len(data_vals)-TAMANO_VENTANA//2, PASO)[:X_wins.shape[0]]
    y_wins = labels[idx]
    dev_wins = devices[idx]
    
    print(f"Ventanas generadas: {X_wins.shape[0]}")
    return X_wins, y_wins, dev_wins, list(le_act.keys())

# ================= FEATURES (GPU) =================
def kernel_features_gpu_safe(tensor_gpu):
    d = tensor_gpu.astype(cp.float32)
    fts = {}
    
    # 1. Stats
    mean = cp.mean(d, axis=1)
    std = cp.std(d, axis=1)
    for i, ax in enumerate(['x', 'y', 'z']):
        fts[f'mean_{ax}'] = mean[:, i]
        fts[f'std_{ax}'] = std[:, i]
    del mean, std

    # 2. Correlaciones
    mu = cp.mean(d, axis=1, keepdims=True)
    d_cent = d - mu
    sig = cp.std(d, axis=1, keepdims=True)
    
    cov_xy = cp.mean(d_cent[:,:,0] * d_cent[:,:,1], axis=1)
    fts['corr_xy'] = cov_xy / (sig[:,0,0] * sig[:,0,1] + 1e-9)
    
    cov_yz = cp.mean(d_cent[:,:,1] * d_cent[:,:,2], axis=1)
    fts['corr_yz'] = cov_yz / (sig[:,0,1] * sig[:,0,2] + 1e-9)
    
    cov_zx = cp.mean(d_cent[:,:,2] * d_cent[:,:,0], axis=1)
    fts['corr_zx'] = cov_zx / (sig[:,0,2] * sig[:,0,0] + 1e-9)
    del mu, d_cent, sig, cov_xy, cov_yz, cov_zx

    # 3. SMA/Energia
    fts['sma'] = cp.mean(cp.sum(cp.abs(d), axis=2), axis=1)
    energy = cp.mean(d**2, axis=1)
    for i, ax in enumerate(['x', 'y', 'z']):
        fts[f'energy_{ax}'] = energy[:, i]
    del energy

    # 4. Jerk
    jerk = cp.diff(d, axis=1)
    jerk_mean = cp.mean(jerk, axis=1)
    for i, ax in enumerate(['x', 'y', 'z']):
        fts[f'jerk_{ax}'] = jerk_mean[:, i]
    del jerk, jerk_mean

    # 5. FFT
    fft_vals = cp.abs(cp.fft.fft(d, axis=1))
    sum_d = cp.sum(d, axis=2)
    fft_sum = cp.abs(cp.fft.fft(sum_d, axis=1))
    
    idx_dom = cp.argmax(fft_sum[:, 1:TAMANO_VENTANA//2], axis=1) + 1
    fts['dom_freq'] = idx_dom.astype(cp.float32)
    del sum_d, fft_sum, idx_dom

    psd = fft_vals**2
    psd_prob = psd / cp.sum(psd, axis=1, keepdims=True)
    ent = -cp.sum(psd_prob * cp.log(psd_prob + 1e-10), axis=1)
    fts['spec_ent'] = cp.mean(ent, axis=1)
    del fft_vals, psd, psd_prob, ent

    return cp.hstack([v.reshape(-1, 1) for k, v in fts.items()])

def extraer_features_por_lotes(tensor_cpu, batch_size=200):
    res_list = []
    total = tensor_cpu.shape[0]
    
    print(f"Procesando {total} ventanas en lotes de {batch_size}...")
    try: cp.fft.config.get_plan_cache().clear()
    except: pass

    for i in range(0, total, batch_size):
        if i % 10000 == 0: print(f"      -> {i}/{total}...", end="\r")
        
        lote_gpu = cp.asarray(tensor_cpu[i:i+batch_size].astype(np.float32))
        feats_gpu = kernel_features_gpu_safe(lote_gpu)
        
        try: res_cpu = feats_gpu.get()
        except: res_cpu = feats_gpu
        
        res_list.append(res_cpu)
        del lote_gpu, feats_gpu
        limpiar_gpu()
        try: cp.fft.config.get_plan_cache().clear()
        except: pass
        
    print(f"      -> 100% Completado.           ")
    return np.vstack(res_list)

# ================= PREDECIR =================
def predecir_hibrido(modelo_gpu, X_test_scaled_cpu, chunk_size=2000):
    preds = []
    total = len(X_test_scaled_cpu)
    
    for i in range(0, total, chunk_size):
        chunk_cpu = X_test_scaled_cpu[i : i + chunk_size]
        chunk_gpu = cp.asarray(chunk_cpu)
        p_gpu = modelo_gpu.predict(chunk_gpu)
        
        try: p_cpu = p_gpu.get()
        except: p_cpu = p_gpu.to_numpy()
        
        preds.append(p_cpu)
        del chunk_gpu, p_gpu
        limpiar_gpu()
        
    return np.concatenate(preds)

# ================= MAIN =================
if __name__ == "__main__":
    limpiar_gpu()
    print("\nINICIANDO EJECUCION FINAL CON GRAFICAS (CPU vs GPU)")
    print(f"   GPU Activa: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    
    resultados_globales = []
    
    for sensor, ruta in SENSORES_A_PROBAR.items():
        print(f"\n" + "="*50)
        print(f"SENSOR: {sensor}")
        print("="*50)
        limpiar_gpu()
        
        # 1. Carga
        try:
            X_raw, y_all, dev_all, clases = cargar_datos_ventaneados(ruta)
            if X_raw is None: continue
        except Exception as e:
            print(f"Error carga: {e}")
            continue

        # 2. Features
        t0 = time.time()
        X_feat = extraer_features_por_lotes(X_raw)
        t_feat = time.time() - t0
        print(f"Features extraidas en {t_feat:.2f}s")
            
        # 3. Split
        df = pd.DataFrame(X_feat)
        df['y'] = y_all
        df['dev'] = dev_all
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['y'])
        
        X_test_np = test_df.drop(['y', 'dev'], axis=1).values.astype(np.float32)
        y_test_true = test_df['y'].values
        
        # 4. Scaler
        print(" Ajustando Escalador Global (CPU)...")
        scaler_cpu = CPU_StandardScaler()
        sample_size = min(len(train_df), 100000)
        X_train_sample = train_df.drop(['y', 'dev'], axis=1).iloc[:sample_size].values.astype(np.float32)
        scaler_cpu.fit(X_train_sample)
        
        X_test_scaled_cpu = scaler_cpu.transform(X_test_np)
        del X_train_sample, X_test_np
        gc.collect()
        
        # 5. Training
        print(" Entrenando Ensamble...")
        preds_matriz = []
        devs_train = train_df['dev'].unique()
        count = 0
        
        t_start = time.time()
        for i, d_code in enumerate(devs_train):
            sub = train_df[train_df['dev'] == d_code]
            if len(sub) > 50:
                print(f"      -> Disp {d_code}...", end="\r")
                X_sub_cpu = sub.drop(['y', 'dev'], axis=1).values.astype(np.float32)
                y_sub_cpu = sub['y'].values.astype(np.int32)
                
                X_sub_scaled_cpu = scaler_cpu.transform(X_sub_cpu)
                X_sub_gpu = cp.asarray(X_sub_scaled_cpu)
                y_sub_gpu = cp.asarray(y_sub_cpu)
                
                # Cache pequeño
                svm = GPU_SVC(kernel='rbf', C=10, gamma='scale', cache_size=200) 
                svm.fit(X_sub_gpu, y_sub_gpu)
                p = predecir_hibrido(svm, X_test_scaled_cpu)
                preds_matriz.append(p)
                count += 1
                
                del svm, X_sub_gpu, y_sub_gpu, X_sub_scaled_cpu, X_sub_cpu
                limpiar_gpu()
        
        t_train = time.time() - t_start
        print(f"\n Modelos: {count}. Tiempo Train: {t_train:.2f}s")
        
        # 6. Resultados y Graficas
        if preds_matriz:
            print(" Votando...")
            matriz = np.array(preds_matriz)
            y_pred, _ = mode(matriz, axis=0, keepdims=False)
            y_pred = y_pred.ravel()
            
            acc = accuracy_score(y_test_true, y_pred)
            print(f" RESULTADO FINAL: {acc*100:.2f}%")
            
            # Guardar resultados
            resultados_globales.append({
                'Sensor': sensor,
                'Tiempo_GPU': t_feat + t_train,
                'Accuracy': acc * 100
            })

            # --- REPORTE Y MATRIZ DE CONFUSION ---
            print("\n   --- REPORTE DETALLADO ---")
            print(classification_report(y_test_true, y_pred, target_names=clases))
            
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_test_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=clases, yticklabels=clases)
            plt.title(f'Matriz de Confusion - {sensor} (GPU)')
            plt.ylabel('Real')
            plt.xlabel('Predicho')
            plt.tight_layout()
            plt.show(block=True) # Pausa para ver la imagen
            
        limpiar_gpu()

    # --- COMPARATIVA FINAL CPU vs GPU ---
    print("\n" + "="*50)
    print("GENERANDO GRAFICAS COMPARATIVAS FINALES")
    print("="*50)
    
    df_res = pd.DataFrame(resultados_globales)
    
    # Mapear los tiempos CPU
    df_res['Tiempo_CPU'] = df_res['Sensor'].map(TIEMPOS_CPU_BASELINE)
    df_res['Speedup'] = df_res['Tiempo_CPU'] / df_res['Tiempo_GPU']
    
    print("\nTABLA DE RESULTADOS UNIFICADA:")
    print(df_res.to_string())
    
    # Configurar graficas
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Analisis de Rendimiento: Mejor CPU vs GPU (RTX 3050)', fontsize=16)
    
    # 1. Tiempos de Ejecucion
    ind = np.arange(len(df_res))
    width = 0.35
    ax1.bar(ind - width/2, df_res['Tiempo_CPU'], width, label='CPU (Mejor Paralelo)', color='salmon')
    ax1.bar(ind + width/2, df_res['Tiempo_GPU'], width, label='GPU (CUDA/RAPIDS)', color='skyblue')
    ax1.set_ylabel('Tiempo Total (segundos)')
    ax1.set_title('Tiempo de Ejecucion (Menos es mejor)')
    ax1.set_xticks(ind)
    ax1.set_xticklabels(df_res['Sensor'], rotation=15)
    ax1.legend()
    # Etiquetas
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1fs', padding=3)
    
    # 2. Speedup (Aceleracion)
    sns.barplot(data=df_res, x='Sensor', y='Speedup', ax=ax2, palette='viridis', hue='Sensor')
    ax2.set_ylabel('Speedup (Veces mas rapido)')
    ax2.set_title('Factor de Aceleracion (GPU / CPU)')
    ax2.axhline(1, color='red', linestyle='--', label='Base (Igualdad)')
    ax2.legend()
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.2fx', padding=3)
    
    plt.tight_layout()
    plt.show(block=True)