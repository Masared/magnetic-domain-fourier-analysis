import numpy as np

def calculate_radial_profile(power_spectrum):
    """
    
    2Dパワースペクトルから動径プロファイル（1D）を計算する関数。
    中心からの各半径の円周上におけるパワーの平均値を返します。

    Args:
        power_spectrum (np.ndarray): 2Dのパワースペクトル画像。
                                     直流成分が中心にあること（fftshift済み）。

    Returns:
        np.ndarray: 動径プロファイル（1次元配列）。インデックスが中心からの半径に対応。
    """
    # 画像の中心座標を計算
    center_y, center_x = np.array(power_spectrum.shape) / 2.0

    # 各ピクセルの中心からの距離を計算するためのグリッドを作成
    y, x = np.indices(power_spectrum.shape)
    
    # 各ピクセルの中心からの距離を計算
    #radii:各ピクセルの中心からの距離の2D配列
    radii = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # 距離を整数に丸めて、各半径のビン（入れ物）として使う
    #小数点以下を切り捨て
    radii_int = radii.astype(int)
    # 各半径ビンに含まれるピクセル数を計算，インデックスが半径に該当，値がピクセル数

    pixel_counts = np.bincount(radii_int.ravel())
    
    # 各半径ビンにおけるパワーの合計値を計算 (重み付きのbincount)
    #スペクトルの画像を1Dに並列し，半径ごとの合計値を算出．radii_intには，半径の情報が入っている．power_sumのindexがその半径のピクセル全ての合計値
    power_sum = np.bincount(radii_int.ravel(), weights=power_spectrum.ravel())

    # 0除算を避けるため、ピクセル数が0のビンは平均も0にする
    radial_profile = np.zeros_like(power_sum)
    # ピクセル数が0より大きいビンのみで平均を計算
    non_zero_counts = pixel_counts > 0
    radial_profile[non_zero_counts] = power_sum[non_zero_counts] / pixel_counts[non_zero_counts]
    
    return radial_profile

# --- 関数の使い方・実行サンプル ---
if __name__ == '__main__':
    # サンプルとして、中心から特定の半径にパワーが集中したパワースペクトルを模擬的に作成
    # (実際には、ここにあなたの `mag` を入れてください)
    y, x = np.indices((256, 256))
    center_y, center_x = 128, 128
    radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    # 半径30と60のあたりに強いパワーを持つ模擬スペクトル
    mock_mag = np.exp(-((radius - 30)**2) / 10) + np.exp(-((radius - 60)**2) / 10)
    mock_mag[center_y, center_x] = 5 # 直流成分も少し足しておく
    
    # 関数を呼び出して動径プロファイルを計算
    features_1d = calculate_radial_profile(mock_mag)
    
    # print("--- 動径プロファイルから得られた特徴量ベクトル ---")
    # print(features_1d)
    print("\n特徴量ベクトルの長さ:", len(features_1d))

    # # 結果をプロットして確認
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(mock_mag, cmap='viridis')
    # plt.title('Mock Power Spectrum (`mag`)')
    
    # plt.subplot(1, 2, 2)
    # plt.plot(features_1d)
    # plt.title('Radial Profile (Feature Vector)')
    # plt.xlabel('Radius (Spatial Frequency)')
    # plt.ylabel('Average Power')
    # plt.grid(True)
    # plt.show()