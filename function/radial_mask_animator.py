import os
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from function.mask_pattern import apply_center_mask_and_ifft


def create_radius_sweep_gif(f_shift, radii_list, gif_filename="frequency_sweep.gif", save_path=None, fps=10):
    """
    マスク半径を変化させながらフィルタリングを行い、結果をGIFアニメーションとして保存する。
    保存先のディレクトリを指定可能。

    Args:
        f_shift (np.ndarray): フーリエ変換・シフト済みの2D画像。
        radii_list (list or range): 試行する半径のリスト。
        gif_filename (str, optional): 保存するGIFファイルの名前。
        save_path (str, optional): 保存先のディレクトリパス。指定しない場合はカレントディレクトリ。
        fps (int, optional): GIFのフレームレート。
    """
    if save_path:
        # 保存先ディレクトリが存在しない場合は作成する
        os.makedirs(save_path, exist_ok=True)
        # 保存パスとファイル名を結合して、完全な出力パスを作成
        full_output_path = os.path.join(save_path, gif_filename)
    else:
        # save_pathが指定されていない場合は、ファイル名のみ（カレントディレクトリに保存）
        full_output_path = gif_filename
    images_for_gif = []
    print(f"GIF作成を開始します。ファイル名: {gif_filename}")

    # --- ステップ1: 各半径で画像を生成し、リストに保存 ---
    for radius in tqdm(radii_list, desc="フレームを生成中"):
        
        # フィルタリングを実行（プロットは不要なのでFalseのまま）
        filtered_img = apply_center_mask_and_ifft(f_shift, radius)
        
        # --- ステップ2: 画像をPillowで扱える形式に変換し、テキストを描画 ---
        # 画像を0-255の範囲に正規化して、符号なし8ビット整数に変換
        norm_img = ((filtered_img - filtered_img.min()) / 
                    (filtered_img.max() - filtered_img.min()) * 255).astype(np.uint8)
        
        # Numpy配列をPillow画像オブジェクトに変換
        pil_img = Image.fromarray(norm_img).convert('RGB')
        
        # 描画オブジェクトを作成
        draw = ImageDraw.Draw(pil_img)
        
        # テキストを描画（フォントは環境に合わせて調整してください）
        try:
            # 高品質なフォントを読み込む（ファイルパスを指定）
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            # フォントが見つからない場合はデフォルトフォントを使用
            font = ImageFont.load_default()
        
        draw.text((10, 10), f"Radius: {radius}", font=font, fill=(255, 255, 0)) # 黄色い文字
        
        # GIF用のリストに画像を追加（Pillow画像をNumpy配列に戻す）
        images_for_gif.append(np.array(pil_img))

    # --- ステップ3: 画像リストからGIFを生成して保存 ---
    print("すべての画像の処理が完了しました。GIFファイルを保存します...")
    imageio.mimsave(full_output_path, images_for_gif, fps=fps)
    print(f"完了しました！ '{full_output_path}' が保存されました。")