# imports required for CNN
import torch
import time
import os
from torch.utils.data import DataLoader

from latticevision.plotting import plot_example_field, plot_losses

from latticevision.cnn.dataset import make_dataset, DataConfig
from latticevision.cnn.model import ModelConfig, CNN
from latticevision.cnn.train import TrainingConfig, train_model
from latticevision.cnn.eval import eval_model

# imports required for local CNN estimation on STUN data
from latticevision.img2img.dataset import make_dataset as make_img2img_dataset
from latticevision.img2img.dataset import DataConfig as Img2ImgDataConfig
from latticevision.cnn.eval import fast_cnn_field_tiler
from latticevision.plotting import plot_img2img_samples
from latticevision.img2img.eval import eval_model as eval_img2img_model


def test_cnn():
	# set the device you are on to cpu (github servers)
	device = torch.device("cpu")

	# load and create dataset
	dataset_path = "sample_data/CNN_test_data.h5"
	val_size = 0.4
	test_size = 0.5

	data_config = DataConfig(
		file_path=dataset_path,
		fullsidelen=25,
		sidelen=25,
		n_replicates=30,
		n_params=3,
		log_kappa2=True,
		val_size=val_size,
		test_size=test_size,
		random_state=777,
		verbose=True,
	)

	data_dict = make_dataset(data_config)

	train_df = data_dict["train_df"]
	val_df = data_dict["val_df"]
	test_df = data_dict["test_df"]

	# example field plotter
	plot_example_field(
		dataset=train_df,
		config=data_config,
		idx=0,
		model_type="CNN",
		field_color="turbo",
		show=False,
	)

	# put into dataloaders
	n_batch = 2

	train_loader = DataLoader(train_df, batch_size=n_batch, shuffle=True)
	val_loader = DataLoader(val_df, batch_size=n_batch, shuffle=False)
	test_loader = DataLoader(test_df, batch_size=n_batch, shuffle=False)

	for fields, params in train_loader:
		print("Train Loader:")
		print(
			"Fields batch shape: ", fields.shape, "\nParams batch shape: ", params.shape
		)
		break

	for fields, params in val_loader:
		print("Validation Loader:")
		print(
			"Fields batch shape: ", fields.shape, "\nParams batch shape: ", params.shape
		)
		break

	for fields, params in test_loader:
		print("Test Loader:")
		print(
			"Fields batch shape: ", fields.shape, "\nParams batch shape: ", params.shape
		)
		break

	# config = ModelConfig(
	#     sidelen=25,
	# 	in_channels=30,
	# 	out_params=3,
	# 	conv_channels=(64, 128, 256),
	# 	linear_sizes=(500,64),
	#     kernel_sizes = (10,7,5),
	#     padding = 0,
	# )

	# load in tiny model
	config = ModelConfig(
		sidelen=25,
		in_channels=30,
		out_params=3,
		conv_channels=(2, 4),
		linear_sizes=(16, 8),
		kernel_sizes=(11, 10),
		padding=0,
	)

	model = CNN(config).to(device)
	total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Total number of trainable parameters: {total_params}")

	# create directory
	save_path = "tests/_temp/model_wghts/"
	os.makedirs(save_path, exist_ok=True)

	# train model
	start_time = time.time()

	train_config = TrainingConfig(
		model=model,
		device=device,
		train_loader=train_loader,
		val_loader=val_loader,
		train_df=train_df,
		val_df=val_df,
		lr=5e-4,
		n_epochs=2,
		stop_patience=10,
		scheduler_patience=5,
		scheduler_factor=0.5,
		augmentation=True,
		save=True,
		save_directory=save_path,
		savename="cnn_wghts.pth",
		verbose=True,
		normalize=True,
		shuffle=True,
	)

	training_results = train_model(config=train_config)

	print("--- %s seconds ---" % (time.time() - start_time))
	time_train = time.time() - start_time
	print("Training took: ", time_train / 60, " minutes.")

	model = training_results["model"]
	train_losses = training_results["train_losses"]
	val_losses = training_results["val_losses"]

	# test the loss plotting function
	plot_losses(
		train_losses=train_losses,
		val_losses=val_losses,
		show=False,
	)

	# load the model
	model_loaded = CNN(config)
	model_loaded.load_state_dict(torch.load(save_path + "cnn_wghts.pth"))
	model_loaded = model_loaded.to(device)
	model_loaded.eval()

	# test the eval function
	metrics = eval_model(
		model=model,
		device=device,
		config=data_config,
		test_loader=test_loader,
		test_df=test_df,
		plot=True,
		augmentation=True,
		show=False,
	)

	# load in the big STUN data for testing
	dataset_path_i2i = "sample_data/STUN_test_data.h5"
	val_size_i2i = 0.4
	test_size_i2i = 0.5

	data_config_i2i = Img2ImgDataConfig(
		file_path=dataset_path_i2i,
		n_rows=192,
		n_cols=288,
		n_replicates=30,
		n_params=3,
		log_kappa2=True,
		shift_theta=True,
		val_size=val_size_i2i,
		test_size=test_size_i2i,
		random_state=777,
		verbose=True,
	)

	data_dict = make_img2img_dataset(
		config=data_config_i2i,
	)

	test_df_i2i = data_dict["test_df"]

	n_batch = 2
	test_loader_i2i = DataLoader(test_df_i2i, batch_size=n_batch, shuffle=False)
	for fields, params in test_loader_i2i:
		print("Test Loader:")
		print(
			"Fields batch shape: ", fields.shape, "\nParams batch shape: ", params.shape
		)
		break

	# perform cnn local estimation (tiling)
	test_fields = test_df_i2i[:][0]
	test_params = test_df_i2i[:][1]

	outputs = torch.zeros(test_params.size())

	# fast option
	for i in range(len(outputs)):
		output = fast_cnn_field_tiler(
			model=model,
			fields=test_fields[i].unsqueeze(0),
			device=device,
			patch_batch_size=10000,
			verbose=False,
			padding_mode="reflect",
			patch_size=25,
		)
		outputs[i] = output

	metrics = eval_img2img_model(
		model=model,
		config=data_config_i2i,
		device=device,
		test_loader=test_loader_i2i,
		test_df=test_df_i2i,
		plot=True,
		augmentation=False,
		n_pixels=5000,
		cnn_mode=True,
		cnn_results=outputs,
		show=False,
	)

	inds = [0]

	plot_img2img_samples(
		model=model,
		config=data_config_i2i,
		device=device,
		test_df=test_df_i2i,
		indices=inds,
		random_selection=False,
		num_rand_samples=5,
		awght_not_kappa2=True,
		cnn_mode=True,
		cnn_results=outputs,
		show=False,
	)
