import os
import argparse

# =========================
# Project paths
# =========================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(PROJECT_ROOT, 'data')

RAW_DIR = os.path.join(root_dir, 'raw')
MAT_DIR = os.path.join(root_dir, 'mat')
PATCH_DIR = os.path.join(root_dir, 'patch')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')

res_map = {
    '5kb': 5_000,
    '10kb': 10_000,
    '25kb': 25_000,
    '50kb': 50_000,
    '100kb': 100_000,
    '250kb': 250_000,
    '500kb': 500_000,
    '1mb': 1_000_000
}

# You can adjust these splits later if needed.
# For now, this is enough to get the GM12878 bulk pipeline running.
set_dict = {
    'K562_test': [4, 14, 16, 20],
    'NHEK_test': [4, 14, 16, 20],
    'HMEC_test': [4, 14, 16, 20],
    'GM12878_test': [4, 14, 16, 20],

    # training/validation on GM12878
    'train': [1, 3, 5, 7, 8, 9, 11, 13, 15, 17, 18, 19, 21, 22],
    'valid': [2, 6, 10, 12]
}
# 受内存限制，先用已经有的数据进行处理
# set_dict = {
#     'K562_test': [20, 21, 22],
#     'NHEK_test': [20, 21, 22],
#     'HMEC_test': [20, 21, 22],
#     'GM12878_test': [20, 21, 22],
#
#     'train': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
#     'valid': [19, 20, 21, 22]
# }

help_opt = (('--help', '-h'), {
    'action': 'help',
    'help': "Print this help message and exit"
})


def mkdir(out_dir):
    if not os.path.isdir(out_dir):
        print(f'Making directory: {out_dir}')
    os.makedirs(out_dir, exist_ok=True)


def chr_num_str(x):
    start = x.find('chr')
    part = x[start + 3:]
    end = part.find('_')
    return part[:end]


def chr_digit(filename):
    chrn = chr_num_str(os.path.basename(filename))
    if chrn == 'X':
        return 23
    return int(chrn)


def data_read_parser():
    parser = argparse.ArgumentParser(
        description="Read raw data from Rao's Hi-C files.",
        add_help=False
    )
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument(
        '-c', dest='cell_line',
        help='REQUIRED: Cell line folder name under data/raw, e.g. GM12878_primary',
        required=True
    )

    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument(
        '-hr', dest='high_res',
        help='High resolution specified [default: 10kb]',
        default='10kb', choices=res_map.keys()
    )
    misc_args.add_argument(
        '-q', dest='map_quality',
        help='Mapping quality of raw data [default: MAPQGE30]. '
             'Currently only used as optional directory filtering.',
        default='MAPQGE30', choices=['MAPQGE30', 'MAPQG0']
    )
    misc_args.add_argument(
        '-n', dest='norm_file',
        help='Normalization file suffix [default: KRnorm]',
        default='KRnorm', choices=['KRnorm', 'SQRTVCnorm', 'VCnorm']
    )
    parser.add_argument(*help_opt[0], **help_opt[1])
    return parser


def data_down_parser():
    parser = argparse.ArgumentParser(
        description='Downsample high-resolution data',
        add_help=False
    )
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument(
        '-c', dest='cell_line',
        help='REQUIRED: Cell line folder name, e.g. GM12878_primary',
        required=True
    )
    req_args.add_argument(
        '-hr', dest='high_res',
        help='REQUIRED: High resolution specified [example: 10kb]',
        default='10kb', choices=res_map.keys(), required=True
    )
    req_args.add_argument(
        '-lr', dest='low_res',
        help='REQUIRED: Low resolution tag [example: 40kb]',
        default='40kb', required=True
    )
    req_args.add_argument(
        '-r', dest='ratio',
        help='REQUIRED: Downsampling ratio [example: 16]',
        default=16, type=int, required=True
    )
    parser.add_argument(*help_opt[0], **help_opt[1])
    return parser


def data_divider_parser():
    parser = argparse.ArgumentParser(
        description='Generate train/valid/test patch data',
        add_help=False
    )
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument(
        '-c', dest='cell_line',
        help='REQUIRED: Cell line folder name, e.g. GM12878_primary',
        required=True
    )
    req_args.add_argument(
        '-hr', dest='high_res',
        help='REQUIRED: High resolution specified [example: 10kb]',
        default='10kb', choices=res_map.keys(), required=True
    )
    req_args.add_argument(
        '-lr', dest='low_res',
        help='REQUIRED: Low resolution specified [example: 40kb]',
        default='40kb', required=True
    )
    req_args.add_argument(
        '-lrc', dest='lr_cutoff',
        help='REQUIRED: Cutoff for low-resolution maps [example: 100]',
        default=100, type=int, required=True
    )
    req_args.add_argument(
        '-s', dest='dataset',
        help='Dataset split to generate',
        default='train',
        choices=['K562_test', 'NHEK_test', 'HMEC_test', 'GM12878_test', 'train', 'valid']
    )

    hicarn_args = parser.add_argument_group('HiCARN Arguments')
    hicarn_args.add_argument(
        '-chunk', dest='chunk',
        help='Chunk size [default: 40]',
        default=40, type=int, required=True
    )
    hicarn_args.add_argument(
        '-stride', dest='stride',
        help='Stride [default: 40]',
        default=40, type=int, required=True
    )
    hicarn_args.add_argument(
        '-bound', dest='bound',
        help='Distance boundary [default: 201]',
        default=201, type=int, required=True
    )
    hicarn_args.add_argument(
        '-scale', dest='scale',
        help='Pooling scale [default: 1]',
        default=1, type=int, required=True
    )
    hicarn_args.add_argument(
        '-type', dest='pool_type',
        help='Pooling type [default: max]',
        default='max', choices=['max', 'avg']
    )
    parser.add_argument(*help_opt[0], **help_opt[1])
    return parser


def data_predict_parser():
    parser = argparse.ArgumentParser(
        description='Predict data using DiCARN model',
        add_help=False
    )
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument(
        '-c', dest='cell_line',
        help='REQUIRED: Cell line for analysis [example: GM12878_primary]',
        required=True
    )
    req_args.add_argument(
        '-lr', dest='low_res',
        help='REQUIRED: Low resolution specified [example: 40kb]',
        default='40kb', required=True
    )
    req_args.add_argument(
        '-f', dest='file_name',
        help='REQUIRED: Patch file to be enhanced',
        required=True
    )
    req_args.add_argument(
        '-m', dest='model',
        help='OPTIONAL: model name',
        required=False
    )

    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument(
        '-ckpt', dest='checkpoint',
        help='REQUIRED: checkpoint file',
        required=True
    )
    misc_args.add_argument(
        '--cuda', dest='cuda',
        help='Use CUDA or not [default: 0]',
        default=0, type=int
    )
    parser.add_argument(*help_opt[0], **help_opt[1])
    return parser


def train_parser():
    parser = argparse.ArgumentParser(
        description='Train DiCARN on generated bulk Hi-C patches',
        add_help=False
    )
    req_args = parser.add_argument_group('Required Arguments')
    req_args.add_argument(
        '-c', dest='cell_line',
        help='REQUIRED: Cell line folder name, e.g. GM12878_primary',
        required=True
    )
    req_args.add_argument(
        '-hr', dest='high_res',
        help='REQUIRED: High resolution [example: 10kb]',
        default='10kb', choices=res_map.keys(), required=True
    )
    req_args.add_argument(
        '-lr', dest='low_res',
        help='REQUIRED: Low resolution tag [example: 40kb]',
        default='40kb', required=True
    )

    misc_args = parser.add_argument_group('Miscellaneous Arguments')
    misc_args.add_argument('-chunk', dest='chunk', default=40, type=int)
    misc_args.add_argument('-stride', dest='stride', default=40, type=int)
    misc_args.add_argument('-bound', dest='bound', default=201, type=int)
    misc_args.add_argument('-scale', dest='scale', default=1, type=int)
    misc_args.add_argument('-type', dest='pool_type', default='max', choices=['max', 'avg'])
    misc_args.add_argument('-epochs', dest='epochs', default=50, type=int)
    misc_args.add_argument('-bs', dest='batch_size', default=64, type=int)
    misc_args.add_argument('-nw', dest='num_workers', default=4, type=int)
    misc_args.add_argument('--cuda', dest='cuda', default=1, type=int)
    parser.add_argument(*help_opt[0], **help_opt[1])
    return parser
# import os
# import argparse
#
# # the Root directory for all raw and processed data
# # root_dir = 'Data/DNase/target/CH12-LX'  # Example of root directory name
# root_dir = 'dicarn_project_data/Test/K562' # adjust accordingly
#
# res_map = {'5kb': 5_000, '10kb': 10_000, '25kb': 25_000, '50kb': 50_000, '100kb': 100_000, '250kb': 250_000,
#            '500kb': 500_000, '1mb': 1_000_000}
#
# # 'train' and 'valid' can be changed for different train/valid set splitting
# set_dict = {'K562_test': [4, 14, 16, 20],
#             'NHEK_test': [4, 14, 16, 20],
#             'HMEC_test': [4, 14, 16, 20],
#             'CH12-LX_test': [4, 14, 16, 19],
#             'mESC_test': (4, 9, 15, 18),
#             # 'train': [1, 3, 5, 7, 8, 9, 11, 13, 15, 17, 18, 19, 21, 22],
#             'train': [4, 14, 16],
#             'valid': [2, 6, 10, 12],
#             'GM12878_test': (4, 14, 16, 20)}
#
# help_opt = (('--help', '-h'), {
#     'action': 'help',
#     'help': "Print this help message and exit"})
#
#
# def mkdir(out_dir):
#     if not os.path.isdir(out_dir):
#         print(f'Making directory: {out_dir}')
#     os.makedirs(out_dir, exist_ok=True)
#
#
# # chr12_10kb.npz, predict_chr13_40kb.npz
# def chr_num_str(x):
#     start = x.find('chr')
#     part = x[start + 3:]
#     end = part.find('_')
#     return part[:end]
#
#
# def chr_digit(filename):
#     chrn = chr_num_str(os.path.basename(filename))
#     if chrn == 'X':
#         n = 23
#     else:
#         n = int(chrn)
#     return n
#
#
# def data_read_parser():
#     parser = argparse.ArgumentParser(description='Read raw data from Rao\'s Hi-C.', add_help=False)
#     req_args = parser.add_argument_group('Required Arguments')
#     req_args.add_argument('-c', dest='cell_line', help='REQUIRED: Cell line for analysis[example:GM12878]',
#                           required=True)
#
#     misc_args = parser.add_argument_group('Miscellaneous Arguments')
#     misc_args.add_argument('-hr', dest='high_res', help='High resolution specified[default:10kb]',
#                            default='10kb', choices=res_map.keys())
#     misc_args.add_argument('-q', dest='map_quality', help='Mapping quality of raw data[default:MAPQGE30]',
#                            default='MAPQGE30', choices=['MAPQGE30', 'MAPQG0'])
#     misc_args.add_argument('-n', dest='norm_file', help='The normalization file for raw data[default:KRnorm]',
#                            default='KRnorm', choices=['KRnorm', 'SQRTVCnorm', 'VCnorm'])
#     parser.add_argument(*help_opt[0], **help_opt[1])
#
#     return parser
#
#
# def data_down_parser():
#     parser = argparse.ArgumentParser(description='Downsample data from high resolution data', add_help=False)
#     req_args = parser.add_argument_group('Required Arguments')
#     req_args.add_argument('-c', dest='cell_line', help='REQUIRED: Cell line for analysis[example:GM12878]',
#                           required=True)
#     req_args.add_argument('-hr', dest='high_res', help='REQUIRED: High resolution specified[example:10kb]',
#                           default='10kb', choices=res_map.keys(), required=True)
#     req_args.add_argument('-lr', dest='low_res', help='REQUIRED: Low resolution specified[example:40kb]',
#                           default='40kb', required=True)
#     req_args.add_argument('-r', dest='ratio', help='REQUIRED: The ratio of downsampling[example:16]',
#                           default=16, type=int, required=True)
#     parser.add_argument(*help_opt[0], **help_opt[1])
#
#     return parser
#
#
# def data_divider_parser():
#     parser = argparse.ArgumentParser(description='Divide data for train and predict', add_help=False)
#     req_args = parser.add_argument_group('Required Arguments')
#     req_args.add_argument('-c', dest='cell_line', help='REQUIRED: Cell line for analysis[example:GM12878]',
#                           required=True)
#     req_args.add_argument('-hr', dest='high_res', help='REQUIRED: High resolution specified[example:10kb]',
#                           default='10kb', choices=res_map.keys(), required=True)
#     req_args.add_argument('-lr', dest='low_res', help='REQUIRED: Low resolution specified[example:40kb]',
#                           default='40kb', required=True)
#     req_args.add_argument('-lrc', dest='lr_cutoff', help='REQUIRED: cutoff for low resolution maps[example:100]',
#                           default=100, type=int, required=True)
#     req_args.add_argument('-s', dest='dataset', help='REQUIRED: Dataset for train/valid/predict(all)',
#                           default='train', choices=['K562_test', 'NHEK_test', 'DNase_train', 'HMEC_test', 'mESC_test', 'train', 'valid', 'GM12878_test', 'CH12-LX_test'], )
#     hicarn_args = parser.add_argument_group('HiCARN Arguments')
#     hicarn_args.add_argument('-chunk', dest='chunk', help='REQUIRED: chunk size for dividing[example:40]',
#                               default=40, type=int, required=True)
#     hicarn_args.add_argument('-stride', dest='stride', help='REQUIRED: stride for dividing[example:40]',
#                               default=40, type=int, required=True)
#     hicarn_args.add_argument('-bound', dest='bound', help='REQUIRED: distance boundary interested[example:201]',
#                               default=201, type=int, required=True)
#     hicarn_args.add_argument('-scale', dest='scale', help='REQUIRED: Downpooling scale[example:1]',
#                               default=1, type=int, required=True)
#     hicarn_args.add_argument('-type', dest='pool_type', help='OPTIONAL: Downpooling type[default:max]',
#                               default='max', choices=['max', 'avg'])
#     parser.add_argument(*help_opt[0], **help_opt[1])
#
#     return parser
#
#
# def data_predict_parser():
#     parser = argparse.ArgumentParser(description='Predict data using HiCARN model', add_help=False)
#     req_args = parser.add_argument_group('Required Arguments')
#     req_args.add_argument('-c', dest='cell_line', help='REQUIRED: Cell line for analysis[example: GM12878]',
#                           required=True)
#     req_args.add_argument('-lr', dest='low_res', help='REQUIRED: Low resolution specified[example: 40kb]',
#                           default='40kb', required=True)
#     req_args.add_argument('-f', dest='file_name', help='REQUIRED: Matrix file to be enhanced[example: '
#                                                        'hicarn_10kb40kb_c40_s40_b201_nonpool_human_GM12878_test.npz', required=True)
#     req_args.add_argument('-m', dest='model', help='REQUIRED: Choose your model[example: HiCARN_1]', required=False)
#     gan_args = parser.add_argument_group('GAN model Arguments')
#     gan_args.add_argument('-ckpt', dest='checkpoint', help='REQUIRED: Checkpoint file of HiCARN model',
#                           required=True)
#     misc_args = parser.add_argument_group('Miscellaneous Arguments')
#     misc_args.add_argument('--cuda', dest='cuda', help='Whether or not using CUDA[default:1]',
#                            default=0, type=int)
#     parser.add_argument(*help_opt[0], **help_opt[1])
#
#     return parser
