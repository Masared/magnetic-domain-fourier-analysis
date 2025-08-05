import numpy as np
import matplotlib.pyplot as plt

def apply_center_mask_and_ifft(f_shift, radius, plot_result=False):
    """
    フーリエ変換後の画像中心に円形マスクを適用し、逆フーリエ変換を行う。
    オプションで、処理の各段階の画像をプロットする。

    Args:
        f_shift (np.ndarray): フーリエ変換され、直流成分が中心にシフトされた2次元画像。
        radius (int): マスクする円の半径（ピクセル単位）。
        plot_result (bool, optional): Trueの場合、処理過程と結果をプロットする。
                                      デフォルトは False。

    Returns:
        np.ndarray: マスク処理後、逆フーリエ変換された画像（実数値の振幅画像）。
    """
    # --- ステップ1~4は前回と同じ ---
    f_shift_masked = f_shift.copy()
    rows, cols = f_shift.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol)**2 + (y - crow)**2 <= radius**2
    f_shift_masked[mask_area] = 0
    
    # --- ステップ5: 逆フーリエ変換を実行 ---
    f_ishift = np.fft.ifftshift(f_shift_masked)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # --- ステップ6: 結果のプロット (plot_resultがTrueの場合のみ実行) ---
    if plot_result:
        print(f"--- Plotting results for radius: {radius} ---")
        # スペクトルは見やすいように対数スケールに変換します
        spectrum_before = 20*np.log(np.abs(f_shift))
        spectrum_after = 20*np.log(np.abs(f_shift_masked))

        # 3つの画像を並べて表示する設定
        plt.figure(figsize=(18, 6))

        # 1. マスクをかける前のフーリエスペクトル
        plt.subplot(1, 3, 1)
        plt.imshow(spectrum_before, cmap='gray')
        plt.title('1. Spectrum (Before Masking)')
        plt.axis('off')

        # 2. マスクをかけた後のフーリエスペクトル
        plt.subplot(1, 3, 2)
        plt.imshow(spectrum_after, cmap='gray')
        plt.title(f'2. Spectrum (After Masking, r={radius})')
        plt.axis('off')
        
        # 3. 逆フーリエ変換後の最終的な画像
        plt.subplot(1, 3, 3)
        plt.imshow(img_back, cmap='gray')
        plt.title('3. Resulting Image (iFFT)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    return img_back