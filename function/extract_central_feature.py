import numpy as np
import matplotlib.pyplot as plt
def extract_unique_central_features(power_spectrum, window_size=11, include_dc=True):
    """
    パワースペクトルの中心領域から、点対称性を考慮してユニークな係数のみを抽出する。
    具体的には、中心領域の上半分（中央行の右半分を含む）を返す。これは、フリーデルの法則（Friedel's Law）から明らか。

    Args:
        power_spectrum (np.ndarray): 2Dパワースペクトル画像（fftshift済み）。
        window_size (int): 切り出す正方形領域の1辺のピクセル数（奇数を推奨）。
        include_dc (bool): Trueの場合、直流成分を含める。デフォルトはTrue。

    Returns:
        np.ndarray: ユニークな係数を1次元に並べた特徴量ベクトル。
    """
    center_y, center_x = np.array(power_spectrum.shape) // 2#中心座標を計算
    half_size = window_size // 2
    
    # 中心領域をスライス
    central_region = power_spectrum[
        center_y - half_size : center_y + half_size + 1,
        center_x - half_size : center_x + half_size + 1
    ]
    
    unique_features = []
    # 中心領域の上半分（中央行まで）をループ
    for y in range(half_size + 1):
        if y < half_size:
            # 上半分の行はすべての列を追加
            row = central_region[y, :]
            unique_features.extend(row)
        else: # y == half_size (中央行)
            # 中央行は、中心から右側の列のみ追加
            row = central_region[y, half_size:]
            unique_features.extend(row)
            
    features = np.array(unique_features)

    # include_dcがFalseの場合、直流成分（新しいベクトルの先頭要素）を削除
    if not include_dc:
        # この方法では直流成分は常に先頭に来る
        features = np.delete(features, 0)
        
    return features

# --- 関数の使い方・実行サンプル ---
if __name__ == '__main__':
    # -1, 1の2値画像からパワースペクトルを計算
    image_size = 256
    input_image = np.ones((image_size, image_size))
    for i in range(0, image_size, 20):
        input_image[:, i:i+10] = -1
    mag = np.abs(np.fft.fftshift(np.fft.fft2(input_image)))

    window_size = 11

    # --- 元の方法 ---
    features_original = extract_central_features(mag, window_size=window_size, include_dc=True)
    print("--- 元の方法（冗長性あり）---")
    print(f"特徴量ベクトルの長さ: {len(features_original)} (={window_size}x{window_size})")

    # --- 新しい方法 ---
    features_unique = extract_unique_central_features(mag, window_size=window_size, include_dc=True)
    print("\n--- 新しい方法（対称性を考慮し、冗長性を排除）---")
    print(f"特徴量ベクトルの長さ: {len(features_unique)}")

    # 理論値: (N*N - 1)/2 + 1  (N=11の場合, (121-1)/2 + 1 = 61)
    theoretical_len = (window_size**2 - 1) / 2 + 1
    print(f"理論的な長さ: {int(theoretical_len)}")
