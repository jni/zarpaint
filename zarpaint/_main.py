import argparse
import ast
import sys

from ._zarpaint import correct_labels


parser = argparse.ArgumentParser(
        'zarpaint',
        description='Paint segmentations directly to '
                    'on-disk/remote zarr arrays',
        )
parser.add_argument('image', help='The input image file.')
parser.add_argument('labels', help='The labels file (read/write).')
parser.add_argument('--frame', type=int, help='Load only this frame.')
parser.add_argument(
        '-s', '--scale',
        type=ast.literal_eval,
        default=(4, 1, 1),
        help='Scale factors.',
        )
parser.add_argument(
        '-c', '--channel',
        type=int,
        default=2,
        help='Which channel to load.',
        )


def main():
    args = parser.parse_args(sys.argv[1:])
    correct_labels(
            args.image, args.labels,
            time_index=args.frame,
            scale=args.scale,
            c=args.channel,
            )
