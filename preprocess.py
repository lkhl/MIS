import argparse
import os
import os.path as osp

from mis.data import ParallelPreprocessor, Preprocessor


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data-root', '-d', type=str, required=True, help='Root directory for the SBD dataset')
    parser.add_argument(
        '--out-dir',
        '-o',
        type=str,
        default='./data/proposals/sbd',
        help='Output directory for the preprocessed data')
    parser.add_argument(
        '--model-size',
        '-m',
        type=str,
        choices=('small', 'base', 'large', 'giant'),
        default='small',
        help='Model size of the ViT')
    parser.add_argument(
        '--patch-size',
        '-p',
        type=int,
        choices=(8, 14, 16),
        default=8,
        help='Patch size of the ViT')
    parser.add_argument(
        '--n-featurizing-workers',
        type=int,
        default=2,
        help='Number of workers for featurizing. Set to 0 to disable parallel processing')
    parser.add_argument(
        '--n-merging-workers',
        type=int,
        default=4,
        help='Number of workers for merging. Set to 0 to disable parallel processing')

    return parser.parse_args()


def main():
    args = parse_args()

    img_dir = osp.join(args.data_root, 'img')
    with open(osp.join(args.data_root, 'train.txt'), 'r') as f:
        files = f.readlines()
    files = [osp.join(img_dir, img.strip() + '.jpg') for img in files]

    os.makedirs(args.out_dir, exist_ok=True)

    if args.n_featurizing_workers < 1 or args.n_merging_workers < 1:
        preprocessor = Preprocessor()
    else:
        preprocessor = ParallelPreprocessor()

    preprocessor(
        files=files,
        out_dir=args.out_dir,
        model_size=args.model_size,
        patch_size=args.patch_size,
        n_featurizing_workers=args.n_featurizing_workers,
        n_merging_workers=args.n_merging_workers)


if __name__ == '__main__':
    main()
