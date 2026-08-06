import cv2
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

TILE = 128
OVERLAP = 16
STEP = TILE - OVERLAP

def haar_1d(src):
    """1D Haar на массиве длины TILE, возвращает низкие и высокие"""
    n = len(src)
    low = np.zeros(n//2)
    high = np.zeros(n//2)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for i in range(n//2):
        a = src[2*i]
        b = src[2*i+1]
        low[i] = (a + b) * inv_sqrt2
        high[i] = (a - b) * inv_sqrt2
    return np.concatenate([low, high])

def dwt_2d_haar(tile):
    """2D Haar на тайле"""
    n = tile.shape[0]
    # по строкам
    tmp = np.zeros_like(tile)
    for i in range(n):
        tmp[i, :] = haar_1d(tile[i, :])
    # по столбцам (транспонируем, применяем 1D, транспонируем обратно)
    tmp = tmp.T
    for i in range(n):
        tmp[i, :] = haar_1d(tmp[i, :])
    return tmp.T

def extract_features_from_tile(gray_tile):
    """Возвращает массив из 16 фичей для одного тайла"""
    coeffs = dwt_2d_haar(gray_tile)
    n2 = TILE // 2
    bands = [
        (0, n2, 0, n2),
        (0, n2, n2, TILE),
        (n2, TILE, 0, n2),
        (n2, TILE, n2, TILE)
    ]
    features = []
    for (r1, r2, c1, c2) in bands:
        sub = coeffs[r1:r2, c1:c2].flatten()
        mean = np.mean(sub)
        var = np.var(sub)
        energy = np.sum(sub**2)
        sq = sub**2
        p = sq / (energy + 1e-12)
        entropy = -np.sum(p * np.log(p + 1e-12))
        features.extend([mean, var, energy, entropy])
    return np.array(features)

def main():
    image_dir = "images"
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.bmp', '.png', '.jpg'))]
    
    all_features = []
    all_info = []  

    for img_file in image_files:
        img = cv2.imread(os.path.join(image_dir, img_file))
        if img is None:
            continue
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        h, w = gray.shape
        for y in range(0, h - TILE + 1, STEP):
            for x in range(0, w - TILE + 1, STEP):
                tile = gray[y:y+TILE, x:x+TILE]
                feats = extract_features_from_tile(tile.astype(np.float32))
                all_features.append(feats)
                all_info.append((img_file, x, y))
    
    # Сохраняем CSV
    cols = [f'f{i}' for i in range(16)]
    df = pd.DataFrame(all_features, columns=cols)
    df['image'] = [info[0] for info in all_info]
    df['x'] = [info[1] for info in all_info]
    df['y'] = [info[2] for info in all_info]
    df.to_csv('features.csv', index=False)
    print(f"Saved {len(df)} tiles to features.csv")

    df = pd.read_csv("features.csv")
    cols = [f'f{i}' for i in range(16)]
    # ---- K-Means ----
    X = df[cols].values
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)
    print("Cluster means:")
    print(df.groupby('cluster')[cols].mean())

    # Scatter plot: f2 (ll_energy) vs f14 (hh_energy)
    plt.figure(figsize=(8,6))
    for cl in range(4):
        subset = df[df['cluster'] == cl]
        plt.scatter(subset['f2'], subset['f14'], label=f'cluster {cl}', s=1)
    plt.xlabel('f2 (ll_energy)')
    plt.ylabel('f14 (hh_energy)')
    plt.title('K-Means clustering (k=2)')
    plt.legend()
    plt.savefig('scatter.png')
    plt.show()

if __name__ == '__main__':
    main()