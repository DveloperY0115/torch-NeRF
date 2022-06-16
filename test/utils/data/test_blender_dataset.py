"""A test script for 'NeRFBlenderDataset'."""

import sys

sys.path.append(".")
sys.path.append("..")

from src.utils.data.blender_dataset import NeRFBlenderDataset


def main():
    """The entry point of test."""

    # initialize dataset
    root_path = "data/nerf_synthetic/lego"
    dataset = NeRFBlenderDataset(root_path)

    # inspect attributes
    print(f"Number of samples: {len(dataset)}")
    print(
        "Dataset attributes (height, width, focal length): "
        f"{dataset.img_height}, {dataset.img_width}, {dataset.focal_length}"
    )


if __name__ == "__main__":
    main()
