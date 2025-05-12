from itertools import product
import torch
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader

import time
import os

# directly import from latticevision library in directory
# from latticevision.device import set_device
from latticevision.img2img.dataset import (
	make_dataset,
	DataConfig,
	no_transform,
	polar_transform,
)
from latticevision.plotting import plot_example_field, plot_losses, plot_img2img_samples
from latticevision.img2img.base import (
	ModelConfig,
	PosEmbed,
	NullPosEmbed,
	LearnedPosEmbed,
	SinusoidalPosEmbed,
	RotaryPosEmbed,
)
from latticevision.img2img import TransUNet, UNet, ViT
from latticevision.img2img.train import train_model, TrainingConfig
from latticevision.img2img.eval import eval_model


@pytest.mark.parametrize(
	("model_cls", "pos_embed_cls"),
	product(
		[UNet, TransUNet, ViT],
		[NullPosEmbed, LearnedPosEmbed, SinusoidalPosEmbed, RotaryPosEmbed],
	),
)
def test_stun(model_cls: nn.Module, pos_embed_cls: PosEmbed):
	# set the device you are on
	device = torch.device("cpu")
	transform_funcs = [no_transform, polar_transform]

	# load and create dataset
	dataset_path = "sample_data/STUN_test_data.h5"
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
		idx=0,
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

	for fields, params in train_loader:
		print("Train Loader:")
		print(
			"Fields batch shape: ",
			fields.shape,
			"\nParams batch shape: ",
			params.shape,
		)
		break

	for fields, params in val_loader:
		print("Val Loader:")
		print(
			"Fields batch shape: ",
			fields.shape,
			"\nParams batch shape: ",
			params.shape,
		)
		break

	for fields, params in test_loader:
		print("Test Loader:")
		print(
			"Fields batch shape: ",
			fields.shape,
			"\nParams batch shape: ",
			params.shape,
		)
		break

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
	model = model_cls(config)
	model = model.to(device)
	total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Total number of trainable parameters: {total_params}")

	# train the model
	start_time = time.time()

	# create directory
	save_path = "tests/_temp/model_wghts/"
	os.makedirs(save_path, exist_ok=True)

	train_config = TrainingConfig(
		model=model,
		device=device,
		train_loader=train_loader,
		val_loader=val_loader,
		train_df=train_df,
		val_df=val_df,
		lr=8e-5,
		n_epochs=2,
		stop_patience=10,
		scheduler_patience=5,
		scheduler_factor=0.5,
		augmentation=True,
		save=True,
		save_directory=save_path,
		savename="stun_wghts.pth",
		verbose=True,
		normalize=True,
		shuffle=True,
	)

	training_results = train_model(config=train_config)

	print("--- %s seconds ---" % (time.time() - start_time))
	time_train = time.time() - start_time
	print("Training took: ", time_train / 60, " minutes.")

	# save training results
	model = training_results["model"]
	train_losses = training_results["train_losses"]
	baseline_losses = training_results["baseline_losses"]
	val_losses = training_results["val_losses"]

	# test the loss plotting function
	plot_losses(
		train_losses=train_losses,
		val_losses=val_losses,
		base_losses=baseline_losses,
		show=False,
	)

	# load the model weights
	model_loaded = model_cls(config)
	model_loaded.load_state_dict(torch.load(save_path + "stun_wghts.pth"))
	model_loaded = model_loaded.to(device)
	model_loaded.eval()

	# evaluate model performance
	metrics = eval_model(
		model=model_loaded,
		config=data_config,
		device=device,
		test_loader=test_loader,
		test_df=test_df,
		plot=True,
		augmentation=True,
		n_pixels=5000,
		show=False,
	)

	print(metrics)

	# test the sample output plotting function
	inds = [0]

	plot_img2img_samples(
		model=model,
		config=data_config,
		device=device,
		test_df=test_df,
		indices=inds,
		random_selection=False,
		awght_not_kappa2=False,
		num_rand_samples=1,
		show=False,
	)
