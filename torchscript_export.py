import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import List

import torch

# Add 'core', which contains the imports for MonSter, to the module search path.
sys.path.append("core")
from monster import Monster
from utils.utils import InputPadder


class WrappedMonSter(torch.nn.Module):
    """ A wrapper around the original MonSter implementation to simplify the TorchScript interface. """

    def __init__(self, monster: torch.nn.Module):
        """ The constructor of WrappedMonSter.

        Args:
            monster (torch.nn.Module): The scriptable MonSter module.
        """
        super().__init__()
        self.monster = monster

    def forward(
        self,
        left_image: torch.Tensor,
        right_image: torch.Tensor
    ) -> torch.Tensor:
        """ Given the left and right stereo pair, predict and return the disparity.

        Args:
            left_image (torch.Tensor): The left image of the stereo pair.
            right_image (torch.Tensor): The right image of the stereo pair.

        Returns:
            The predicted disparity as a PyTorch tensor.
        """

        left_image = left_image.unsqueeze(0)
        right_image = right_image.unsqueeze(0)

        # Now pad to make divisible by 32.
        padder = InputPadder(left_image.shape, divis_by=32)
        left_image, right_image = padder.pad(inputs=[left_image, right_image])

        # Run the monster model.
        disp = self.monster(left_image, right_image, iters=24, test_mode=True)

        # Remove the padding.
        disp = padder.unpad(disp)

        disp = torch.squeeze(disp[:, 0, :, :])

        return disp


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

    parser.add_argument(
        "restore_ckpt",
        help="Path to the trained model's restore checkpoint",
        type=Path,
        nargs="?",
        default=Path("pretrained") / 'mix_all.pth'
    )

    parser.add_argument(
        "script_path",
        help="Path to save the torchscript model",
        type=Path,
        nargs="?",
        default=Path("output") / 'monster-mix-script.pt'
    )

    return parser.parse_args(arg_list)


def main() -> None:
    args = parse_args()

    # TODO: Attempt to download missing PyTorch models.
    assert os.path.exists(args.restore_ckpt), f"Model checkpoint file does not exist: {args.restore_ckpt}"
    # Note: MonSter relies on the depth anything v2 pytorch model. We assume it is located in the same dir.
    depth_anything_path = args.restore_ckpt.with_name("depth_anything_v2_vitl.pth")
    assert os.path.exists(depth_anything_path), f"Model checkpoint file does not exist: {depth_anything_path}"

    checkpoint = torch.load(args.restore_ckpt, map_location="cpu", weights_only=True)

    # Load the model.
    model = Monster()
    model.load_state_dict(checkpoint, strict=True)

    # Wrap the model to create a simplified interface.
    monster = WrappedMonSter(model)
    monster = monster.eval()

    # Export the MonSter PyTorch model to TorchScript and save it to the disk.
    try:
        script = torch.jit.script(monster)

        # Create missing directories to save the TorchScript model.
        split_out_path = args.script_path.parts
        if len(split_out_path) > 1:
            os.makedirs(os.path.join(*split_out_path[:-1]), exist_ok=True)
        # Save the model
        script.save(args.script_path)

        print("Successfully exported the model to TorchScript")
        print(f"File saved to {args.script_path}")

    except Exception as e:
        print("Error: Failed to export the model to TorchScript")
        print(e)


if __name__ == "__main__":
    main()
