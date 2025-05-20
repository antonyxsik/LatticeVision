import torch.nn as nn
import pytest
import torch
import os 
from torch.utils.data import DataLoader
from itertools import product

# directly import from latticevision library in directory
# from latticevision.device import set_device
from latticevision.img2img.dataset import (
	make_dataset,
	no_transform,
	polar_transform,
	DataConfig,
)
from latticevision.plotting import plot_example_field
from latticevision.img2img.base import (
	ModelConfig,
	PosEmbed,
	NullPosEmbed,
	LearnedPosEmbed,
	SinusoidalPosEmbed,
	RotaryPosEmbed,
)
from latticevision.img2img import TransUNet, UNet, ViT


# check for data locally, if not found, download from gdrive
DATA_DIR = "data"
DATA_FILE = "I2I_sample_data.h5"
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)
GDRIVE_ID = "1Hz1aRc49sBy0d74iwkfxu_djzwZsEW39"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_ID}"


if not os.path.isdir(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.isfile(DATA_PATH):
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "Please install gdown (`pip install gdown`) so the test can auto-download the data."
        )
    print(f"{DATA_PATH} not found. Downloading from Google Drive...")
    gdown.download(GDRIVE_URL, DATA_PATH, quiet=False)


def test_load_data():
	# load and create dataset
	transform_funcs = [no_transform, polar_transform]

	# load and create dataset
	dataset_path = DATA_PATH

	val_size = 0.4
	test_size = 0.5

	data_config = DataConfig(
		file_path=dataset_path,
		n_rows=192,
		n_cols=288,
		n_replicates=30,
		n_params=3,
		transform_function=transform_funcs[0],
		log_kappa2=True,
		shift_theta=True,
		val_size=val_size,
		test_size=test_size,
		random_state=777,
		verbose=True,
	)

	data_dict = make_dataset(
		config=data_config,
	)

	train_df = data_dict["train_df"]
	val_df = data_dict["val_df"]
	test_df = data_dict["test_df"]

	# test the example field plotter function
	plot_example_field(
		dataset=train_df,
		config=data_config,
		idx=0,  # 27 is nice looking
		model_type="STUN",
		field_color="turbo",
		param1_color="viridis",
		param2_color="viridis",
		param3_color="viridis",
		show=False,
	)

	# create dataloaders
	n_batch = 2

	train_loader = DataLoader(train_df, batch_size=n_batch, shuffle=True)
	val_loader = DataLoader(val_df, batch_size=n_batch, shuffle=False)
	test_loader = DataLoader(test_df, batch_size=n_batch, shuffle=False)

	assert len(train_loader) > 0
	assert len(val_loader) > 0
	assert len(test_loader) > 0


@pytest.mark.parametrize(
	"model_cls",
	[TransUNet, UNet, ViT],
)
def test_create_model(model_cls: nn.Module):
	# create config
	if model_cls == UNet:
		config = ModelConfig(
			embed_dim=16,
			enc_block_channels=(2, 4),
			group_norm_groups=(1, 2),
		)
	elif model_cls == TransUNet:
		config = ModelConfig(
			patch_size_h=2,
			patch_size_w=2,
			embed_dim=16,
			enc_block_channels=(2, 4),
			group_norm_groups=(1, 2),
			num_layers=2,
			num_heads=2,
			mlp_dim=32,
		)
	elif model_cls == ViT:
		config = ModelConfig(
			patch_size_h=16,
			patch_size_w=16,
			embed_dim=16,
			num_layers=2,
			num_heads=2,
			mlp_dim=32,
		)
	else:
		raise ValueError("Invalid model class")

	# create model with specific config, send to device and count params
	model = model_cls(config)
	total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Total number of trainable parameters: {total_params}")


@pytest.mark.parametrize(
	("model_cls", "pos_embed_cls", "batch_size"),
	product(
		[TransUNet, ViT],
		[NullPosEmbed, LearnedPosEmbed, SinusoidalPosEmbed, RotaryPosEmbed],
		[1, 2, 3, 4, 5],
	),
)
def test_pos_embeds(model_cls: nn.Module, pos_embed_cls: PosEmbed, batch_size: int):
	# create config
	if model_cls == UNet:
		config = ModelConfig(
			embed_dim=16,
			enc_block_channels=(2, 4),
			group_norm_groups=(1, 2),
			# pos_embed_cls=pos_embed_cls,
		)
	elif model_cls == TransUNet:
		config = ModelConfig(
			patch_size_h=2,
			patch_size_w=2,
			embed_dim=16,
			enc_block_channels=(2, 4),
			group_norm_groups=(1, 2),
			num_layers=2,
			num_heads=2,
			mlp_dim=32,
			pos_embed_cls=pos_embed_cls,
		)
	elif model_cls == ViT:
		config = ModelConfig(
			patch_size_h=16,
			patch_size_w=16,
			embed_dim=16,
			num_layers=2,
			num_heads=2,
			mlp_dim=32,
			pos_embed_cls=pos_embed_cls,
		)
	else:
		raise ValueError("Invalid model class")

	# create model with specific config, send to device and count params
	model: nn.Module = model_cls(config)
	total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Total number of trainable parameters: {total_params}")

	x_rand = torch.randn(batch_size, config.in_channels, 192, 288)
	# a single forward pass
	model.eval()
	out = model(x_rand)
	assert not out.isnan().any()

	model.train()
	with torch.enable_grad():
		out = model(x_rand)
		assert not out.isnan().any()

		loss = out.sum()
		loss.backward()
