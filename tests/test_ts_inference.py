import os
from pathlib import Path
from packaging import version

import numpy
import torch


def create_test_image(
    fill: numpy.uint8 = 127,
    shape: tuple = (3, 960, 1280),
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """ Create a test image for testing the torchscript MonSter model.

    Args:
        fill    :   A fill value for all the tensor elements.
        shape   :   The dimensions of the image as [C, H, W].
        device  :   The device on which the tensor should be loaded.

    Returns:
        A test image as a torch tensor.
    """

    img = torch.full(shape, fill, dtype=torch.uint8)
    img = img.float().to(device)
    return img


def test_monster_mix_inference():
    # Turn off deprecated nvfuser if Torch version is old.
    # Displays multiple warnings otherwise.
    if version.parse(torch.__version__) < version.parse("2.2.0"):    
        torch._C._jit_set_nvfuser_enabled(False)

    with torch.no_grad():
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # Note: For this test, we assume the TorchScript model is in a fixed dir.
        model_path = Path('output') / 'monster-mix-script.pt'
        assert os.path.exists(model_path), f"TorchScript model file not found at {model_path}"

        model = torch.jit.load(model_path, map_location=device)
        model.eval()

        # Create test images.
        image_left = create_test_image(device=device)
        image_right = create_test_image(device=device)

        # Run inference three times on the same image set.
        for _ in range(3):
            disparity = model(image_left, image_right)
            assert isinstance(disparity, torch.Tensor), "Model output is not a tensor"
