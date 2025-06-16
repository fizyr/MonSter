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
        "--restore_ckpt", help="restore checkpoint", default="./pretrained/mix_all.pth"
    )
    parser.add_argument("--output_directory", help="directory to save output")
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

    # Convert the MonSter model to 'torch.jit.script' and save to disk.
    script = torch.jit.script(monster)
    script.save(os.path.join(cli_args.output_directory, "monster_mix_script.pt"))


if __name__ == "__main__":
    main()
