import sys
import os
import time
import numpy as np

from Utils.io import compactM, divide, pooling
from Arg_Parser import *


def is_valid_npz(npz_path):
    """Check whether an npz file exists and contains required keys."""
    if not os.path.exists(npz_path):
        return False
    try:
        data = np.load(npz_path, allow_pickle=True)
        ok = ('data' in data) and ('target' in data) and ('inds' in data)
        data.close()
        return ok
    except Exception:
        return False


def carn_divider(
    n,
    high_file,
    down_file,
    scale=1,
    pool_type='max',
    chunk=40,
    stride=40,
    bound=201,
    lr_cutoff=100,
    hr_cutoff=255
):
    hic_data = np.load(high_file, allow_pickle=True)
    down_data = np.load(down_file, allow_pickle=True)

    compact_idx = hic_data['compact']
    full_size = hic_data['hic'].shape[0]

    # compact matrices
    hic = compactM(hic_data['hic'], compact_idx)
    down_hic = compactM(down_data['hic'], compact_idx)

    # clamp
    hic = np.minimum(hr_cutoff, hic)
    down_hic = np.minimum(lr_cutoff, down_hic)

    # rescale
    hic = hic / np.max(hic) if np.max(hic) > 0 else hic
    down_hic = down_hic / lr_cutoff

    # divide and pool
    div_dhic, div_inds = divide(down_hic, n, chunk, stride, bound)
    div_dhic = pooling(div_dhic, scale, pool_type=pool_type, verbose=False).numpy()

    div_hhic, _ = divide(hic, n, chunk, stride, bound, verbose=False)

    return n, div_dhic, div_hhic, div_inds, compact_idx, full_size


if __name__ == '__main__':
    args = data_divider_parser().parse_args(sys.argv[1:])

    cell_line = args.cell_line
    high_res = args.high_res
    low_res = args.low_res
    lr_cutoff = args.lr_cutoff
    dataset = args.dataset

    chunk = args.chunk
    stride = args.stride
    bound = args.bound
    scale = args.scale
    pool_type = args.pool_type

    chr_list = set_dict[dataset]
    postfix = dataset
    pool_str = 'nonpool' if scale == 1 else f'{pool_type}pool{scale}'

    print(f'Going to read {high_res} and {low_res} data, then divide matrices with {pool_str}')
    print(f'Cell line: {cell_line}')
    print(f'Dataset split: {dataset}')
    print(f'Chromosomes requested: {chr_list}')

    data_dir = os.path.join(MAT_DIR, cell_line)
    out_dir = os.path.join(PATCH_DIR, cell_line)
    mkdir(out_dir)

    filename = f'hicarn_{high_res}{low_res}_c{chunk}_s{stride}_b{bound}_{pool_str}_{postfix}.npz'
    hicarn_file = os.path.join(out_dir, filename)

    
    if is_valid_npz(hicarn_file):
        print(f'[SKIP] patch file already exists and looks valid: {hicarn_file}')
        sys.exit(0)

   
    if os.path.exists(hicarn_file):
        print(f'[REMOVE] broken/incomplete patch file: {hicarn_file}')
        os.remove(hicarn_file)

    start = time.time()

    results = []
    used_chr = []

    print('[INFO] Running in SERIAL mode (no multiprocessing)')

    for n in chr_list:
        high_file = os.path.join(data_dir, f'chr{n}_{high_res}.npz')
        down_file = os.path.join(data_dir, f'chr{n}_{low_res}.npz')

        if not os.path.exists(high_file):
            print(f'[MISS] skip chr{n}: high-resolution file not found: {high_file}')
            continue
        if not os.path.exists(down_file):
            print(f'[MISS] skip chr{n}: downsampled file not found: {down_file}')
            continue

        print(f'[START] chr{n} | high={os.path.basename(high_file)} | low={os.path.basename(down_file)}')

        kwargs = {
            'scale': scale,
            'pool_type': pool_type,
            'chunk': chunk,
            'stride': stride,
            'bound': bound,
            'lr_cutoff': lr_cutoff
        }

        try:
            res = carn_divider(n, high_file, down_file, **kwargs)
            results.append(res)
            used_chr.append(n)
            print(f'[DONE]  chr{n}')
        except Exception as e:
            print(f'[FAIL]  chr{n} error={e}')

    if len(results) == 0:
        raise RuntimeError('No chromosome patches were generated. Please check available HR/LR npz files.')

    print(f'Used chromosomes: {used_chr}')

    data = np.concatenate([r[1] for r in results], axis=0)
    target = np.concatenate([r[2] for r in results], axis=0)
    inds = np.concatenate([r[3] for r in results], axis=0)
    compacts = {r[0]: r[4] for r in results}
    sizes = {r[0]: r[5] for r in results}

    np.savez_compressed(
        hicarn_file,
        data=data,
        target=target,
        inds=inds,
        compacts=compacts,
        sizes=sizes
    )
    print('Saving file:', hicarn_file)
    print(f'All serial DiCARN patch generation done. Running cost is {(time.time() - start) / 60:.1f} min.')
