import sys
import os
import time
import multiprocessing
import numpy as np

from Utils.io import readcoo2mat
from Arg_Parser import *


def is_valid_npz(npz_path):
    """Check whether an existing npz file is readable and contains required keys."""
    if not os.path.exists(npz_path):
        return False
    try:
        data = np.load(npz_path, allow_pickle=True)
        ok = ('hic' in data) and ('compact' in data)
        data.close()
        return ok
    except Exception:
        return False


def read_data(data_file, norm_file, out_dir, resolution):
    filename = os.path.basename(data_file).split('.')[0] + '.npz'
    out_file = os.path.join(out_dir, filename)

    # skip completed files
    if is_valid_npz(out_file):
        print(f'[SKIP] already exists and looks valid: {out_file}', flush=True)
        return

    # if broken/incomplete file exists, remove it first
    if os.path.exists(out_file):
        print(f'[REMOVE] broken/incomplete output: {out_file}', flush=True)
        try:
            os.remove(out_file)
        except Exception as e:
            print(f'[ERROR] cannot remove {out_file}: {e}', flush=True)
            return

    print(f'[START] data={os.path.basename(data_file)} | norm={os.path.basename(norm_file)}', flush=True)

    try:
        hic, idx = readcoo2mat(data_file, norm_file, resolution)
        np.savez_compressed(out_file, hic=hic, compact=idx)
        print(f'[DONE]  saved: {out_file}', flush=True)
    except Exception as e:
        print(f'[FAIL]  data={data_file} norm={norm_file} error={e}', flush=True)


if __name__ == '__main__':
    args = data_read_parser().parse_args(sys.argv[1:])

    cell_line = args.cell_line
    resolution = args.high_res
    map_quality = args.map_quality
    postfix = [args.norm_file, 'RAWobserved']

   
    cpu_count = multiprocessing.cpu_count()
    pool_num = min(1, cpu_count)

    raw_dir = os.path.join(RAW_DIR, cell_line)

    norm_files = []
    data_files = []

    for root, dirs, files in os.walk(raw_dir):
        if len(files) == 0:
            continue

        # only requested resolution
        if resolution not in root:
            continue

        # enforce map quality if present in path
        if map_quality not in root:
            continue

        for f in files:
            if f.endswith(postfix[0]):
                norm_files.append(os.path.join(root, f))
            elif f.endswith(postfix[1]):
                data_files.append(os.path.join(root, f))

    norm_files = sorted(norm_files)
    data_files = sorted(data_files)

    out_dir = os.path.join(MAT_DIR, cell_line)
    mkdir(out_dir)

    print(f'raw_dir: {raw_dir}')
    print(f'Start reading data, there are {len(norm_files)} files ({resolution}).')
    print(f'Output directory: {out_dir}')
    print(f'Using process_num={pool_num}')

    if len(norm_files) == 0 or len(data_files) == 0:
        print('No matching input files found.')
        sys.exit(1)

    if len(norm_files) != len(data_files):
        print(f'[WARN] number mismatch: norm_files={len(norm_files)}, data_files={len(data_files)}')
        print('Will continue using sorted zip pairs.')

    for d, n in zip(data_files, norm_files):
        print(f'PAIR -> data: {os.path.basename(d)} | norm: {os.path.basename(n)}')

    start = time.time()

    pool = multiprocessing.Pool(processes=pool_num, maxtasksperchild=1)
    for data_fn, norm_fn in zip(data_files, norm_files):
        pool.apply_async(read_data, (data_fn, norm_fn, out_dir, res_map[resolution]))

    pool.close()
    pool.join()

    print(f'All reading processes done. Running cost is {(time.time() - start) / 60:.1f} min.')