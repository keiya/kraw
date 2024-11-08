import os
import argparse
import rawpy
import numpy as np
import tifffile

def extract_raw_pixel_values(file_path, output_path, bright=1.0, exp_shift=1.0, monochrome=None):
    # RAWファイルを読み込む
    with rawpy.imread(file_path) as raw:
        # デモザイク処理を行わずに、half_size=Trueで2x2の平均を取る
        rgb = raw.postprocess(
            use_camera_wb=True,
            output_bps=16,
            half_size=True,
            demosaic_algorithm=None,
            no_auto_bright=True,
            output_color=rawpy.ColorSpace.Adobe,
            bright=bright,  # 明るさのスケーリング
            exp_shift=exp_shift  # 露出シフト
        )
        
        # 正規化
        rgb = rgb.astype(np.float32)
        rgb /= np.max(rgb)
        
        # モノクロ出力モード
        if monochrome:
            r_ratio, g_ratio, b_ratio = monochrome
            gray = (rgb[..., 0] * r_ratio + rgb[..., 1] * g_ratio + rgb[..., 2] * b_ratio)
            rgb = np.stack([gray, gray, gray], axis=-1)
        
    tifffile.imwrite(output_path, rgb)
    
def main():
    # コマンドライン引数のパーサーを設定
    parser = argparse.ArgumentParser(description='Extract RAW pixel values and save as TIFF')
    parser.add_argument('input_file', help='Path to the input RAW file')
    parser.add_argument('output_dir', help='Directory to save the output TIFF file')
    parser.add_argument('--bright', type=float, default=1.0, help='Brightness scaling factor')
    parser.add_argument('--exp_shift', type=float, default=1.0, help='Exposure shift factor')
    parser.add_argument('--monochrome', type=str, help='RGB mix ratios for monochrome output, e.g., "30:50:20"')
    
    # 引数を解析
    args = parser.parse_args()
    
    # モノクロ比率を解析
    monochrome_ratios = None
    if args.monochrome:
        monochrome_ratios = list(map(float, args.monochrome.split(':')))
        total = sum(monochrome_ratios)
        monochrome_ratios = [r / total for r in monochrome_ratios]  # Normalize to sum to 1
    
    # 入力ファイル名から出力ファイル名を生成
    input_filename = os.path.basename(args.input_file)
    output_filename = input_filename + '.tiff'
    output_path = os.path.join(args.output_dir, output_filename)
    
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # RAW画像を処理
    extract_raw_pixel_values(args.input_file, output_path, 
                              bright=args.bright, 
                              exp_shift=args.exp_shift,
                              monochrome=monochrome_ratios)
    
    # 結果を確認
    print(f"TIFF file saved at {output_path}")

if __name__ == '__main__':
    main()
