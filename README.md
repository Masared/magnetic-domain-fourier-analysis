# 磁区パターン画像フーリエ解析ツール (Magnetic Domain Fourier Analysis)

このリポジトリは、磁区パターン画像から空間的特徴量および周波数領域（フーリエ）特徴量を抽出するためのPythonツールキットです。

## 概要

磁気力顕微鏡（MFM）などで得られる磁区パターン画像は、材料の磁気的特性を理解する上で重要です。このツールは、これらの画像から定量的な特徴量を抽出し、材料科学研究におけるデータ駆動型のアプローチを支援することを目的としています。

画像内の個々の磁区の形状（空間領域）と、画像全体の持つパターンの周期性（周波数領域）の両面から分析を行います。

## 主な機能

* **周波数領域特徴量の抽出**:
    * 2次元高速フーリエ変換（FFT）によるパワースペクトルの計算
    * 回転不変な特徴量である**動径プロファイル**の計算 (`calculate_radial_profile`)
    * 低周波数領域の係数を、**対称性を考慮して**効率的に抽出 (`extract_unique_central_features`)

## セットアップ方法

#### 1. リポジトリをクローン

まず、このリポジトリをローカルマシンにクローンします。
```bash
git clone [https://github.com/Masared/magnetic-domain-fourier-analysis.git](https://github.com/Masared/magnetic-domain-fourier-analysis.git)
cd magnetic-domain-fourier-analysis
