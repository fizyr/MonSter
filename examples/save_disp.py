import os
from pathlib import Path
from packaging import version

import torch
import numpy as np
from PIL import Image


def load_image(im_path: Path) -> torch.Tensor:
    """Utility to load an image file as a torch tensor.

    Args:
        im_path (pathlib.Path): The path to the image file on the system.

    Returns:
        A torch tensor of the image on the cuda device.
    """

    img = Image.open(im_path)
    img = np.array(img).astype(np.uint8)
    assert img.ndim == 3, f"Expected 3D tensor, got shape {img.shape}"

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img = img.to("cuda")

    return img


def main():
    # Turn off deprecated nvfuser if Torch version is old.
    # Displays multiple warnings otherwise.
    if version.parse(torch.__version__) < version.parse("2.2.0"):
        torch._C._jit_set_nvfuser_enabled(False)

    with torch.no_grad():
        model_path = Path("output") / "monster-mix-script.pt"
        assert os.path.exists(model_path), f"Model file not found at {model_path}"

        model = torch.jit.load(model_path, map_location="cpu")
        model.eval()

        print("TorchScript model loaded")

        image_left_path = Path("input") / "left" / "sample_l.png"
        image_right_path = Path("input") / "right" / "sample_r.png"

        assert os.path.exists(image_left_path), (
            f"Left image not found at {image_left_path}"
        )
        assert os.path.exists(image_right_path), (
            f"Right image not found at {image_right_path}"
        )

        image_left = load_image(image_left_path)
        image_right = load_image(image_right_path)

        print("Stereo image pair loaded")

        # Predict the disparity for a given stereo image pair.
        disparity = model(image_left, image_right)

        disparity = disparity.cpu().numpy()

        # Scaling by 256 done to minimize precision loss when saving an image.
        # TODO: Normalize values in the range [0, 2^16)
        disparity_image = np.round(disparity * 256).astype(np.uint16)
        # Save the disparity image.
        output_path = Path("output") / "disparity.png"
        Image.fromarray(disparity_image).save(output_path)

        print(f"Image saved to {output_path}")


if __name__ == "__main__":
    main()
