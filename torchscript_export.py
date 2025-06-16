import os

import sys

sys.path.append("core")

import argparse
import torch
from monster import Monster
from utils.utils import InputPadder


class ScriptableMonSter(torch.nn.Module):
    def __init__(self, monster: torch.nn.Module):
        super().__init__()
        self.monster = monster

    def forward(
        self, left_image: torch.Tensor, right_image: torch.Tensor
    ) -> torch.Tensor:
        # Now pad to make divisible by 32
        padder = InputPadder(left_image.shape, divis_by=32)
        left_image, right_image = padder.pad(inputs=[left_image, right_image])

        # Run the monster model
        disp = self.monster(left_image, right_image, iters=24, test_mode=True)

        # Remove padding
        disp = padder.unpad(disp)

        return disp.squeeze()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--restore_ckpt",
        help="restore checkpoint",
        default=os.path.join(".", "pretrained", "mix_all.pth"),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="detect offending (TorchScript incompatible) graph nodes",
    )
    parser.add_argument(
        "--output_directory",
        help="directory to save output",
        default=os.path.join(".", "output"),
    )
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()

    # Load the model.
    model = Monster()

    assert os.path.exists(cli_args.restore_ckpt)
    checkpoint = torch.load(
        cli_args.restore_ckpt, map_location="cpu", weights_only=True
    )

    model.load_state_dict(checkpoint, strict=True)

    # # Wrap the model in a scriptable MonSter model.
    monster = ScriptableMonSter(model)
    monster = monster.eval()

    # Scripting a module fails on first error.
    # It is possible that there are more errors in the module that failed scripting.
    if cli_args.validate:
        named_modules_list = list(monster.named_modules())
        seen_module_class_names = set()

        valid: bool = True

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

        exit(0)

    # Convert the MonSter PyTorch model to TorchScript and save to disk.
    try:
        script = torch.jit.script(monster)
        script.save(os.path.join(cli_args.output_directory, "monster_mix_all.pt"))
        print("Model exported to TorchScript")
    except Exception as e:
        print("❌ Failed to export the model to TorchScript")
        print("If submodules are TorchScript incompatible, run using --validate, fix errors, and repeat")
        print(e)


if __name__ == "__main__":
    main()
