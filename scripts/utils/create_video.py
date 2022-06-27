"""Creates a mp4 video file from images in a directory."""

import argparse
import os

import imageio.v2 as imageio
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str, required=True)
parser.add_argument("--vid_title", type=str, required=True)
args = parser.parse_args()


def main():
    files = [os.path.join(args.img_dir, file) for file in sorted(os.listdir(args.img_dir))]
    imgs = [imageio.imread(file) for file in files]

    w = imageio.get_writer(
        f"{args.vid_title}.mp4",
        format="FFMPEG",
        mode="I",
        fps=24,
    )

    for img in imgs:
        w.append_data(img)
    w.close()


if __name__ == "__main__":
    main()
