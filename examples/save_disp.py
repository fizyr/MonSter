import os
import sys
from typing import List
from pathlib import Path
from packaging import version
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace

import torch
import numpy as np
from PIL import Image


def load_image(im_path: Path, device: str = "cpu") -> torch.Tensor:
    """Utility to load an image file as a torch tensor.

    Args:
        im_path (pathlib.Path): The path to the image file on the system.
        device (str)    : The device on which the tensor should be loaded.

    Returns:
        A torch tensor of the image on the specified device.
    """

    img = Image.open(im_path)
    img = np.array(img).astype(np.uint8)
    assert img.ndim == 3, f"Expected 3D tensor, got shape {img.shape}"

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img = img.to(device)

    return img


def parse_args(arg_list: List[str] = sys.argv[1:]) -> Namespace:
    """ Parse the command line arguments.

	Args:
		arg_list: The list of arguments to be parsed.

	Returns:
		The parsed command line arguments.
	"""

    parser = ArgumentParser(
        description='Script for exporting MonSter to a torchscript module.',
        formatter_class=ArgumentDefaultsHelpFormatter,
		allow_abbrev=False
    )

    parser.add_argument("model",            help="Path to the torchscript model", type=Path)
    parser.add_argument("left_image",       help="Path to the left image in the stereo pair", type=Path)
    parser.add_argument("right_image",      help="Path to the right image in the stereo pair", type=Path)
    parser.add_argument("disparity_path",   help="Path to save the disparity image to", type=Path)

    return parser.parse_args(arg_list)


def main() -> None:
    args = parse_args()

    assert os.path.exists(args.model), f"Given model file does not exist: {args.model}"
    assert os.path.exists(args.left_image), (f"Left image path does not exist: {args.left_image}")
    assert os.path.exists(args.right_image), (f"Right image path does not exist: {args.right_image}")

    # Turn off deprecated nvfuser if Torch version is old.
    # Displays warning messages otherwise.
    if version.parse(torch.__version__) < version.parse("2.2.0"):
        torch._C._jit_set_nvfuser_enabled(False)

    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = torch.jit.load(args.model, map_location=device)
        model.eval()

        print("TorchScript model loaded")

        image_left = load_image(args.left_image, device)
        image_right = load_image(args.right_image, device)

        print("Stereo image pair loaded")

        # Note: Use two warm up runs in production.

        # Predict the disparity for a given stereo image pair.
        disparity = model(image_left, image_right)
        disparity = disparity.cpu().numpy()

        # Scaling by 256 done to minimize precision loss when saving an image.
        # TODO: Normalize values in the range [0, 2^16)
        disparity_image = np.round(disparity * 256).astype(np.uint16)

        # Create missing directories to save the disparity image.
        split_out_path = args.disparity_path.parts
        if len(split_out_path) > 1:
            os.makedirs(os.path.join(*split_out_path[:-1]), exist_ok=True)

        # Save the disparity image.
        Image.fromarray(disparity_image).save(args.disparity_path)
        print(f"Image saved to {args.disparity_path}")


if __name__ == "__main__":
    main()
