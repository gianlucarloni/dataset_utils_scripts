import os
import csv
import argparse
import cv2
from PIL import Image
import numpy as np


def apply_clahe(img: Image.Image) -> Image.Image:
    r"""Normalize image using CLAHE algorithm

    Args:
        img (Image): Input image.

    Returns:
        Normalized image
    """
    img_array = np.array(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for idx in range(3):
        # Per channel normalization
        clahe_img: np.ndarray = clahe.apply(img_array[..., idx])
        if np.amax(clahe_img) > np.amin(clahe_img):
            # Min-max normalization
            clahe_img = (
                (
                    (clahe_img - np.amin(clahe_img))
                    / (np.amax(clahe_img) - np.amin(clahe_img))
                )
                * 255
            ).astype(np.uint8)
        img_array[..., idx] = clahe_img[..., 0]
    return Image.fromarray(img_array)


def get_mass_bbox(mass_mask: Image.Image) -> tuple[int, int, int, int]:
    r"""Returns the (x,y) coordinates of the bounding center, followed by width and height of the box

    Args:
        mass_mask (Image): Image containing the binary mask of the mass.
    """
    mask_array = np.array(mass_mask)
    x_min, x_max = mask_array.shape[1], 0
    for x in range(mask_array.shape[1]):
        if np.max(mask_array[:, x]) > 0:
            x_min = min(x_min, x)
            x_max = x
    y_min, y_max = mask_array.shape[0], 0
    for y in range(mask_array.shape[0]):
        if np.max(mask_array[y, :]) > 0:
            y_min = min(y_min, y)
            y_max = y
    width = x_max - x_min
    height = y_max - y_min
    assert (
        width > 0 and height > 0
    ), f"Invalid bounding box dimensions: width={width}, height={height}"
    return x_min + width // 2, y_min + height // 2, width, height


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop masses and save as png.")
    parser.add_argument(
        "--csv",
        required=True,
        type=str,
        help="csv file containing image paths and labels",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=str,
        help="path to output directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "test"],
        help="target split (either train or test)",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=1000,
        help="size of the crop (in pixels). Default: 1000",
    )
    parser.add_argument(
        "--clahe",
        help="apply CLAHE preprocessing to images",
        action="store_true",
    )

    args = parser.parse_args()

    # Open and read CSV
    with open(args.csv) as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for entry in csv_reader:
            # Temporary
            if "_CC_" in entry["dirname"]:
                continue

            benign = entry["pathology"].startswith(
                "BENIGN"
            )  # Includes BENIGN and BENIGN_WITHOUT_CALLBACK

            # Seek original image
            img_path = os.path.join(
                "full_images",
                entry["dirname"][:-2] + ".png",  # Cut mass index from entry name
            )
            if not os.path.isfile(img_path):
                raise ValueError(f"Could not find image {img_path}")
            img = Image.open(img_path)

            # Seek mass mask
            mask_path = os.path.join("png_masks", args.split, entry["dirname"] + ".png")
            if not os.path.isfile(mask_path):
                raise ValueError(f"Could not find image {mask_path}")
            mask = Image.open(mask_path)

            # Resize if necessary
            if img.size != mask.size:
                print(
                    f"Mismatching size between image and mask for {img_path}. "
                    f"Should be {img.size} but found {mask.size}"
                )
                mask = mask.resize(img.size)
                assert img.size == mask.size

            # Normalize original image if necessary
            if args.clahe:
                img = apply_clahe(img)

            # Get mass center based on mass mask
            x, y, w, h = get_mass_bbox(mask)
            assert (
                w < args.crop_size and h < args.crop_size
            ), f"Crop size {args.crop_size} too small to encapsulate bounding box of size {w} x {h}"
            # Adjust center of the crop if necessary, based on image boundaries
            x_min = x - (args.crop_size // 2)
            x_max = x + (args.crop_size // 2)
            y_min = y - (args.crop_size // 2)
            y_max = y + (args.crop_size // 2)
            if x_min < 0:
                x_max -= x_min
                x_min = 0
            if x_max > img.width:
                x_min -= x_max - img.width
                x_max = img.width
            if y_min < 0:
                y_max -= y_min
                y_min = 0
            if y_max > img.height:
                y_min -= y_max - img.height
                y_max = img.height
            assert (
                x_min >= 0
                and y_min >= 0
                and x_max <= img.width
                and y_max <= img.height
                and (x_max - x_min == args.crop_size)
                and (y_max - y_min == args.crop_size)
            )
            cropped_img = img.crop((x_min, y_min, x_max, y_max))

            # Save image
            output_dir = os.path.join(
                args.output, args.split, "benign" if benign else "malignant"
            )
            os.makedirs(output_dir, exist_ok=True)
            cropped_img.save(os.path.join(output_dir, entry["dirname"] + ".png"))
