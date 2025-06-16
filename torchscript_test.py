import os
import time
from packaging import version

import torch
import numpy as np
from PIL import Image


def load_image(imfile):
    """Utility function to load image files as tensors"""
    img = Image.open(imfile)
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to("cuda")


def main():
    if version.parse(torch.__version__) < version.parse("2.2.0"):
        torch._C._jit_set_nvfuser_enabled(False)

    with torch.no_grad():
        model_path = os.path.join('output', 'monster_mix_all.pt')
        model = torch.jit.load(model_path, map_location="cuda")
        model.eval()

        # Load the left and right image pair.
        image1 = load_image(os.path.join('input', 'left', 'sample_l.png'))
        image2 = load_image(os.path.join('input', 'right', 'sample_r.png'))

        # Two warm-up runs.
        # Note: The second run take the longest.
        print("Running warm-ups")

        start_time = time.time()
        for _ in range(2):
            disparity = model(image1, image2)
            torch.cuda.synchronize()
        end_time = time.time()
        warm_up_time = end_time - start_time

        print(f"Warm-up took {warm_up_time:.4f} seconds\n")

        # Running inference four times on the same image set for demonstration.
        for _ in range(4):
            start_time = time.time()
            disparity = model(image1, image2)
            torch.cuda.synchronize()
            end_time = time.time()

            # Use `disparity` for further processing.
            disparity = disparity.cpu()
            disparity = disparity.numpy()
            
            inference_time = end_time - start_time
            print(f"Inference took {inference_time:.4f} seconds")

            # Scaling by 256 done to minimize precision loss when saving an image.
            disparity_image = np.round(disparity * 256).astype(np.uint16)
            # Save the disparity image.
            output_path = os.path.join('output', 'disparity.png')
            Image.fromarray(disparity_image).save(output_path)


if __name__ == '__main__':
    main()
