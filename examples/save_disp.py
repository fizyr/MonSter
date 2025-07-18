import sys
from typing import List
from pathlib import Path
from packaging import version
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace

import torch
import numpy as np
from PIL import Image


def load_image(image_path: Path, device: str = "cpu") -> torch.Tensor:
    """ Utility to load an image file as a torch tensor.

    Args:
        image_path  : The path to the image file on the system.
        device      : The device on which the tensor should be loaded.

    Returns:
        A torch tensor of the image, placed on the specified device.
    """

    img = Image.open(image_path)
    img = np.array(img).astype(np.uint8)
    assert img.ndim == 3, f"Expected 3D tensor, got shape {img.shape}"

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img = img.to(device)

    return img


def parse_args(arg_list: List[str] = sys.argv[1:]) -> Namespace:
    """ Parse the command line arguments.
    
    Args:
		arg_list:   The list of arguments to be parsed.

    Returns:
        The parsed command line arguments.
    """

    parser = ArgumentParser(
        description='Script for exporting MonSter to a torchscript module.',
        formatter_class=ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )

    # Positional arguments.
    parser.add_argument("model",            type=Path,  help="Path to the torchscript model.")
    parser.add_argument("left_image",       type=Path,  help="Path to the left image in the stereo pair.")
    parser.add_argument("right_image",      type=Path,  help="Path to the right image in the stereo pair.")
    parser.add_argument("disparity_path",   type=Path,  help="Path to save the disparity image to.")
    # Optional arguments.
    parser.add_argument("--device",         type=str,   choices=["cpu", "cuda"],    default="cpu",  help="Path to the torchscript model.")

    return parser.parse_args(arg_list)


def main() -> None:
    args = parse_args()

    assert args.model.is_file(), f"Given model file does not exist: {args.model}"
    assert args.left_image.is_file(), (f"Left image path does not exist: {args.left_image}")
    assert args.right_image.is_file(), (f"Right image path does not exist: {args.right_image}")

    if args.device == "cuda":
        assert torch.cuda.is_available(), "Device type cuda is not available"

    # Turn off deprecated nvfuser if Torch version is old.
    # Displays warning messages otherwise.
    if version.parse(torch.__version__) < version.parse("2.2.0"):
        torch._C._jit_set_nvfuser_enabled(False)

    with torch.no_grad():
        model = torch.jit.load(args.model, map_location=args.device)
        model.eval()

        print("TorchScript model loaded")

        image_left = load_image(args.left_image, args.device)
        image_right = load_image(args.right_image, args.device)

        print("Stereo image pair loaded")

        # Note: Use two warm up runs in a production environment.

        # Predict the disparity for a given stereo image pair.
        disparity = model(image_left, image_right)
        disparity = disparity.cpu().numpy()

        # Scaling by 256 done to minimize precision loss when saving an image.
        # TODO: Normalize values in the range [0, 2^16)
        disparity_image = np.round(disparity * 256).astype(np.uint16)
        # TODO: Save color images for visualization.

        # Save the disparity image.
        args.disparity_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(disparity_image).save(args.disparity_path)
        print(f"Image saved to {args.disparity_path}")


if __name__ == "__main__":
    main()
