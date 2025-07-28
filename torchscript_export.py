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
	"""	A wrapper around the original MonSter implementation to simplify the TorchScript interface. """

	def __init__(self, monster: torch.nn.Module) -> None:
		""" The constructor of WrappedMonSter.

		Args:
			monster	:	The scriptable MonSter module.
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
			left_image	:	The left image of the stereo pair.
			right_image	:	The right image of the stereo pair.

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
	"""	Parse the command line arguments.

	Args:
		arg_list:	The list of arguments to be parsed.

	Returns:
		The parsed command line arguments.
	"""

	parser = ArgumentParser(
		description='Script for exporting MonSter to a torchscript module.',
		formatter_class=ArgumentDefaultsHelpFormatter,
		allow_abbrev=False,
	)

	parser.add_argument("--weights",		type=Path,	default=Path("pretrained") / 'mix_all.pth',			help="Path to the trained model's weights.")
	parser.add_argument("--script_path",	type=Path,	default=Path("output") / 'monster-mix-script.pt',	help="Path to save the torchscript model to.")

	return parser.parse_args(arg_list)


def main() -> None:
	args = parse_args()

	# TODO: Attempt to download missing PyTorch models.

	assert args.weights.is_file(), f"Model weights file does not exist: {args.weights}"
	# Note: MonSter relies on the depth anything v2 pytorch model. We assume it is located in the same dir.
	depth_anything_path = args.weights.with_name("depth_anything_v2_vitl.pth")
	assert depth_anything_path.is_file(), f"Model weights file does not exist: {depth_anything_path}"

	state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)

	# Load the model.
	model = Monster()
	model.load_state_dict(state_dict, strict=True)

	# Wrap the model to create a simplified interface.
	monster = WrappedMonSter(model)
	monster = monster.eval()

	# Export the MonSter PyTorch model to TorchScript and save it to the disk.
	try:
		script = torch.jit.script(monster)

		# Save the model
		args.script_path.parent.mkdir(parents=True, exist_ok=True)
		script.save(args.script_path)

		print("Successfully exported the model to TorchScript")
		print(f"File saved to {args.script_path}")

	except Exception as e:
		print(f"Error: Failed to export the model to TorchScript:\n{e}")


if __name__ == "__main__":
	main()
