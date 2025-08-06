import os
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def create_hpf_sweep_gif(
    image,
    gif_filename="hpf_sweep.gif",
    save_path=None,
    fps=15,
    num_frames=60,
    radii_list=None,
    left_panel_content="spectrum"
):
    """
    左側に選択された画像、右側にハイパスフィルタ適用後の画像を並べたGIFを生成する。

    Args:
        image (np.ndarray): 入力となる256x256のグレースケール画像。
        gif_filename (str, optional): 保存するGIFファイルの名前。
        save_path (str, optional): 保存先のディレクトリパス。
        fps (int, optional): GIFのフレームレート。
        num_frames (int, optional): 自動生成する場合のGIFの総フレーム数。
        radii_list (list or range, optional): 試行するマスク半径のリスト。指定しない場合は自動生成される。
        left_panel_content (str, optional): 左パネルに表示する内容を選択。
                                            'spectrum': マスク適用後の周波数スペクトル (デフォルト)
                                            'mask':     適用するハイパスフィルタのマスク
                                            'original': 元の画像
    """
    # --- 0. 引数のチェックと準備 ---
    if image.ndim != 2 or image.shape[0] != 256 or image.shape[1] != 256:
        print(f"警告: 入力画像は256x256のグレースケール画像を想定しています。現在の形: {image.shape}")

    valid_contents = ['spectrum', 'mask', 'original']
    if left_panel_content not in valid_contents:
        print(f"警告: 'left_panel_content'には{valid_contents}のいずれかを指定してください。'{left_panel_content}'は無効なため、デフォルトの'spectrum'を使用します。")
        left_panel_content = 'spectrum'
    
    height, width = image.shape
    
    if radii_list is None:
        print(f"半径リストが指定されていないため、{num_frames}フレームで自動生成します。")
        max_radius = int(np.sqrt((height / 2)**2 + (width / 2)**2))
        radii_list = np.linspace(0, max_radius, num_frames, dtype=int)
        radii_list = sorted(list(set(radii_list)))

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        full_output_path = os.path.join(save_path, gif_filename)
    else:
        full_output_path = gif_filename

    images_for_gif = []
    print(f"GIF作成を開始します。出力先: {full_output_path}")
    print(f"左パネルの表示内容: {left_panel_content}")

    # --- 1. 元画像のフーリエ変換と、必要な画像の事前準備 ---
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)

    # 'original' モード用に元の画像をPILイメージとして準備
    if left_panel_content == 'original':
        pil_original = Image.fromarray(image).convert('L')
    
    # --- 2. メインループ: 各半径でフレームを生成 ---
    for radius in tqdm(radii_list, desc="フレームを生成中"):
        # マスクを適用した周波数スペクトルを計算
        f_shift_masked = f_shift.copy()
        rows, cols = f_shift.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        y, x = np.ogrid[:rows, :cols]
        # 中心からの距離が半径以下の領域(True)を計算 (これが除去される部分)
        low_pass_area = (x - ccol)**2 + (y - crow)**2 <= radius**2
        f_shift_masked[low_pass_area] = 0
        
        # --- 左パネル用の画像を生成 ---
        if left_panel_content == 'spectrum':
            spectrum_img = np.log(1 + np.abs(f_shift_masked))
            spec_min, spec_max = spectrum_img.min(), spectrum_img.max()
            if spec_max - spec_min > 0:
                spectrum_norm = ((spectrum_img - spec_min) / (spec_max - spec_min) * 255).astype(np.uint8)
            else:
                spectrum_norm = np.zeros_like(spectrum_img, dtype=np.uint8)
            pil_left_panel = Image.fromarray(spectrum_norm).convert('L')
        
        elif left_panel_content == 'mask':
            # ハイパスフィルタのマスクを画像化 (通過域を白、遮断域を黒)
            hpf_mask = (~low_pass_area).astype(np.uint8) * 255
            pil_left_panel = Image.fromarray(hpf_mask).convert('L')

        elif left_panel_content == 'original':
            pil_left_panel = pil_original

        # --- 右パネル用の画像 (フィルタリング後の画像) を生成 ---
        f_ishift = np.fft.ifftshift(f_shift_masked)
        img_back = np.fft.ifft2(f_ishift)
        filtered_img = np.abs(img_back)
        filt_min, filt_max = filtered_img.min(), filtered_img.max()
        if filt_max - filt_min > 0:
            filtered_norm = ((filtered_img - filt_min) / (filt_max - filt_min) * 255).astype(np.uint8)
        else:
            filtered_norm = np.zeros_like(filtered_img, dtype=np.uint8)
        pil_filtered = Image.fromarray(filtered_norm).convert('L')

        # --- 左右の画像を結合して1フレームを作成 ---
        combined_width = width * 2
        combined_img = Image.new('RGB', (combined_width, height))
        combined_img.paste(pil_left_panel, (0, 0))
        combined_img.paste(pil_filtered, (width, 0))

        draw = ImageDraw.Draw(combined_img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        
        draw.text((10, 10), f"Radius: {radius}", font=font, fill=(255, 255, 0))
        images_for_gif.append(np.array(combined_img))

    # --- 3. 画像リストからGIFを生成して保存 ---
    print("すべてのフレームの処理が完了しました。GIFファイルを保存します...")
    imageio.mimsave(full_output_path, images_for_gif, fps=fps)
    print(f"完了しました！ '{full_output_path}' が保存されました。")
    
    return full_output_path