import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import List

import torch

# Add 'core', which contains the imports
# for MonSter to the module search path.
sys.path.append("core")

from monster import Monster
from utils.utils import InputPadder


class ScriptableMonSter(torch.nn.Module):
    def __init__(self, monster: torch.nn.Module):
        super().__init__()
        self.monster = monster

    def forward(
        self,
        left_image: torch.Tensor,
        right_image: torch.Tensor
    ) -> torch.Tensor:

        left_image = left_image.unsqueeze(0)
        right_image = right_image.unsqueeze(0)

        # Now pad to make divisible by 32
        padder = InputPadder(left_image.shape, divis_by=32)
        left_image, right_image = padder.pad(inputs=[left_image, right_image])

        # Run the monster model
        disp = self.monster(left_image, right_image, iters=24, test_mode=True)

        # Remove padding
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

    parser.add_argument("--restore_ckpt", help="restore checkpoint", type=Path, default=Path(".") / "pretrained" / "mix_all.pth")
    parser.add_argument("--validate", action="store_true", help="detect offending (TorchScript incompatible) graph nodes")

    parser.add_argument("--output_directory", help="directory to save output", type=Path, default=Path(".") / "output")

    return parser.parse_args(arg_list)


def main() -> None:
    args = parse_args()

    # Load the model.
    model = Monster()

    assert os.path.exists(args.restore_ckpt)
    checkpoint = torch.load(
        args.restore_ckpt, map_location="cpu", weights_only=True
    )

    model.load_state_dict(checkpoint, strict=True)

    # Wrap the model in a scriptable MonSter model.
    monster = ScriptableMonSter(model)
    monster = monster.eval()

    # Note: The torchscript exporter exits at the first point of failure.
    # The following block of code runs the exporter, individually, on all 1-level deep sub-modules in MonSter to torchscript.

    # Validate whether the model is compatible for TorchScript export.
    if args.validate:
        named_modules_list = list(monster.named_modules())
        seen_module_class_names = set()

        # A flag variable.
        # We assume that the model is compatible for TorchScript export.
        valid: bool = True

        # Validate each module in the model separately.
        for name, module in reversed(named_modules_list):
            class_name = type(module).__name__
            if class_name in seen_module_class_names:
                continue
            seen_module_class_names.add(class_name)

            try:
                torch.jit.script(module)
            except Exception as e:
                valid = False
                print(f"❌ Failed to script: {name} ({type(module)})")
                print(e)

        if valid:
            print("The model is TorchScript compatible")

        # The script was executed for validation only.
        exit(0)

    # Export the MonSter PyTorch model to TorchScript and save it to the disk.
    try:
        script = torch.jit.script(monster)
        script.save(os.path.join(args.output_directory, "monster-mix-script.pt"))
        print("Model exported to TorchScript")
    except Exception as e:
        print("❌ Failed to export the model to TorchScript")
        print("If submodules are TorchScript incompatible, run using --validate, fix errors, and repeat")
        print(e)


if __name__ == "__main__":
    main()
