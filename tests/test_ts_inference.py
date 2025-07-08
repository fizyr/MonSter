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


def test_monster_mix_inference():
    # Turn off deprecated nvfuser if Torch version is old.
    # Displays multiple warnings otherwise.
    if version.parse(torch.__version__) < version.parse("2.2.0"):    
        torch._C._jit_set_nvfuser_enabled(False)

    with torch.no_grad():
        model_path = Path('output') / 'monster-mix-script.pt'
        assert os.path.exists(model_path), f"Model file not found at {model_path}"

        model = torch.jit.load(model_path, map_location="cuda")
        model.eval()

        image_left_path = Path('input') / 'left' / 'sample_l.png'
        image_right_path = Path('input') / 'right' / 'sample_r.png'

        assert os.path.exists(image_left_path), f"Left image not found at {image_left_path}"
        assert os.path.exists(image_right_path), f"Right image not found at {image_right_path}"

        image_left = load_image(image_left_path)
        image_right = load_image(image_right_path)

        # Note: Use two warmup runs in production.

        # Run inference three times on the same image set.
        for _ in range(3):
            disparity = model(image_left, image_right)
            torch.cuda.synchronize()

            assert isinstance(disparity, torch.Tensor), "Model output is not a tensor"
