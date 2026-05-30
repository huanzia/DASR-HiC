import sys
import os
import time
import numpy as np

from Utils.io import downsampling
from Arg_Parser import *


def is_valid_npz(npz_path):
    """Check whether an npz file exists and contains required keys."""
    if not os.path.exists(npz_path):
        return False
    try:
        data = np.load(npz_path, allow_pickle=True)
        ok = ('hic' in data) and ('compact' in data)
        data.close()
        return ok
    except Exception:
        return False


def downsample_one(in_file, low_res, ratio):
    chr_name = os.path.basename(in_file).split('_')[0]
    out_file = os.path.join(os.path.dirname(in_file), f'{chr_name}_{low_res}.npz')

    # skip completed files
    if is_valid_npz(out_file):
        print(f'[SKIP] {chr_name} already exists and looks valid: {out_file}', flush=True)
        return

    # remove broken file first
    if os.path.exists(out_file):
        print(f'[REMOVE] {chr_name} broken/incomplete output: {out_file}', flush=True)
        try:
            os.remove(out_file)
        except Exception as e:
            print(f'[ERROR] cannot remove {out_file}: {e}', flush=True)
            return

    print(f'[START] {chr_name} from {os.path.basename(in_file)}', flush=True)

    try:
        data = np.load(in_file, allow_pickle=True)
        hic = data['hic']
        compact_idx = data['compact']

        down_hic = downsampling(hic, ratio)

        np.savez_compressed(out_file, hic=down_hic, compact=compact_idx, ratio=ratio)
        print(f'[DONE]  {chr_name} saved: {out_file}', flush=True)

    except Exception as e:
        print(f'[FAIL]  {chr_name} error={e}', flush=True)


if __name__ == '__main__':
    args = data_down_parser().parse_args(sys.argv[1:])

    cell_line = args.cell_line
    high_res = args.high_res
    low_res = args.low_res
    ratio = args.ratio

    data_dir = os.path.join(MAT_DIR, cell_line)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f'Matrix directory not found: {data_dir}')

    in_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith('.npz') and f'_{high_res}.npz' in f
    ]
    in_files = sorted(in_files, key=chr_digit)

    print("Input directory :", data_dir)
    print("Number of files :", len(in_files))
    print(f'Generating {low_res} files from {high_res} files by {ratio}x downsampling.')
    print('[INFO] Running in SERIAL mode (no multiprocessing)')

    if len(in_files) == 0:
        raise RuntimeError(f'No "{high_res}" matrix files found in: {data_dir}')

    for f in in_files:
        print('FILE ->', os.path.basename(f))

    start = time.time()

    for file in in_files:
        downsample_one(file, low_res, ratio)

    print(f'All serial downsampling tasks done. Running cost is {(time.time() - start) / 60:.1f} min.')
