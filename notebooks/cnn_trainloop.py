# imports required for CNN
import time
import os
import torch
import numpy as np
import h5py
import pandas as pd
from torch.utils.data import DataLoader

from latticevision.device import set_device

from latticevision.cnn.dataset import make_dataset, DataConfig
from latticevision.cnn.model import ModelConfig, CNN
from latticevision.cnn.train import TrainingConfig, train_model

# imports required for local CNN estimation on STUN data
from latticevision.img2img.dataset import make_dataset as make_img2img_dataset
from latticevision.img2img.dataset import DataConfig as Img2ImgDataConfig
from latticevision.cnn.eval import fast_cnn_field_tiler
from latticevision.img2img.eval import eval_model as eval_img2img_model
from latticevision.seed import set_all_random_seeds
from latticevision.img2img.dataset import polar_transform, no_transform

# make sure folders exist
for d in ("results/model_wghts", "results/metrics", "results/clim_outputs"):
	os.makedirs(d, exist_ok=True)

# set the random seed for reproducibility
set_all_random_seeds(777)

# set the device you are on
device = set_device(
	machine="remote", gpu=True, gpu_id="cuda:0", verbose=True
)  # for remote use

modelsize_list = [9, 17, 25]
num_replicates_list = [1, 5, 15, 30]

# load big i2i data first, depends on num replicates
for num_replicates in num_replicates_list:
	dataset_path_i2i = "data/I2I_data.h5"
	val_size_i2i = 0.1
	test_size_i2i = 0.2

	transform_funcs = [no_transform, polar_transform]

	data_config_i2i = Img2ImgDataConfig(
		file_path=dataset_path_i2i,
		n_rows=192,
		n_cols=288,
		n_replicates=num_replicates,
		transform_function=transform_funcs[0],
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

	del data_dict

	n_batch = 64
	test_loader_i2i = DataLoader(test_df_i2i, batch_size=n_batch, shuffle=False)
	for fields, params in test_loader_i2i:
		print("Test Loader:")
		print(
			"Fields batch shape: ", fields.shape, "\nParams batch shape: ", params.shape
		)
		break

	# loop through cnn models
	for modelsize in modelsize_list:
		dataset_path = "data/CNN_data.h5"
		val_size = 0.1
		test_size = 0.2

		data_config = DataConfig(
			file_path=dataset_path,
			fullsidelen=25,
			sidelen=modelsize,
			n_replicates=num_replicates,
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

		n_batch = 64

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
			print("Validation Loader:")
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

		if modelsize == 9:
			config = ModelConfig(
				sidelen=9,
				in_channels=num_replicates,
				out_params=3,
				conv_channels=(64, 128, 256),
				linear_sizes=(500, 64),
				kernel_sizes=(2, 2, 2),
				padding=0,
			)

		elif modelsize == 17:
			config = ModelConfig(
				sidelen=17,
				in_channels=num_replicates,
				out_params=3,
				conv_channels=(64, 128, 256),
				linear_sizes=(500, 64),
				kernel_sizes=(6, 4, 4),
				padding=0,
			)

		elif modelsize == 25:
			config = ModelConfig(
				sidelen=25,
				in_channels=num_replicates,
				out_params=3,
				conv_channels=(64, 128, 256),
				linear_sizes=(500, 64),
				kernel_sizes=(10, 7, 5),
				padding=0,
			)

		else:
			raise ValueError("Model size not recognized. Please use 9, 17, or 25.")

		print("Running CNN with modelsize: ", modelsize)
		print("Running num_replicates: ", num_replicates)

		model = CNN(config).to(device)
		total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print(f"Total number of trainable parameters: {total_params}")

		meta = {
			"model": "CNN",
			"size": modelsize,
			"reps": num_replicates,
		}
		basename = "_".join(f"{k}{v}" for k, v in meta.items())

		start_time = time.time()

		train_config = TrainingConfig(
			model=model,
			device=device,
			train_loader=train_loader,
			val_loader=val_loader,
			train_df=train_df,
			val_df=val_df,
			lr=1e-4,
			n_epochs=200,
			stop_patience=10,
			scheduler_patience=5,
			scheduler_factor=0.5,
			augmentation=False,
			save=True,
			save_directory="results/model_wghts/",
			savename=f"{basename}.pth",
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

		# eval the model on the i2i data
		test_fields = test_df_i2i[:][0]
		test_params = test_df_i2i[:][1]

		print("Test fields shape: ", test_fields.shape)
		print("Test params shape: ", test_params.shape)

		metrics_start_time = time.time()
		outputs = torch.zeros(test_params.size())
		for i in range(len(outputs)):
			output = fast_cnn_field_tiler(
				model=model,
				fields=test_fields[i].unsqueeze(0),
				device=device,
				patch_batch_size=10000,
				verbose=False,
				padding_mode="reflect",
				patch_size=modelsize,
			)
			outputs[i] = output
		print("--- %s seconds ---" % (time.time() - metrics_start_time))
		time_eval = time.time() - metrics_start_time
		print("Eval took: ", time_eval / 60, " minutes.")

		metrics = eval_img2img_model(
			model=model,
			config=data_config_i2i,
			device=device,
			test_loader=test_loader_i2i,
			test_df=test_df_i2i,
			plot=False,
			augmentation=False,
			n_pixels=5000,
			show=False,
			cnn_mode=True,
			cnn_results=outputs,
		)

		if not isinstance(metrics, pd.DataFrame):
			metrics = pd.DataFrame([metrics])

		metrics_path = "results/metrics/"

		metrics.to_csv(
			metrics_path + f"{basename}_metrics.csv",
			index=False,
		)

		# now we feed in clim fields
		file_path = "data/CESM_LENS_fields.h5"
		with h5py.File(file_path, "r") as f:
			print("Components in the file:", list(f.keys()))

			# extract components
			clim_fields = f["clim_fields"][:]
			clim_fields_norm = f["clim_fields_norm"][:]

			print("clim_fields shape:", clim_fields.shape)

		clim_fields_norm = np.roll(clim_fields_norm, shift=144, axis=-1)
		clim_fields_norm = (
			torch.tensor(clim_fields_norm).unsqueeze(0).float().to(device)
		)

		clim_fields_norm = clim_fields_norm[:, :num_replicates, :, :]
		print("Clim fields shape: ", clim_fields_norm.shape)

		cnn_outputs = torch.zeros(1, 3, 192, 288)
		print(cnn_outputs.shape)

		for i in range(len(cnn_outputs)):
			cnn_output = fast_cnn_field_tiler(
				model=model,
				fields=clim_fields_norm[i].unsqueeze(0),
				device=device,
				patch_batch_size=10000,
				verbose=False,
				padding_mode="reflect",
				patch_size=modelsize,
			)
			cnn_outputs[i] = cnn_output

		print("CNN Clim outputs shape: ", cnn_outputs.shape)

		kappa2_cnn = np.flip(cnn_outputs[0, 0, :, :].detach().cpu().numpy(), axis=0)
		theta_cnn = np.pi / 2 - np.flip(
			cnn_outputs[0, 1, :, :].detach().cpu().numpy(), axis=0
		)
		rho_cnn = np.flip(cnn_outputs[0, 2, :, :].detach().cpu().numpy(), axis=0)
		awght_cnn = np.exp(kappa2_cnn) + 4

		print(
			np.min(awght_cnn),
			np.max(awght_cnn),
			np.min(theta_cnn),
			np.max(theta_cnn),
			np.min(rho_cnn),
			np.max(rho_cnn),
		)

		final = cnn_outputs.detach().cpu().numpy()
		final = np.squeeze(final)
		print(final.shape)

		clim_result_path = "results/clim_outputs/"

		with h5py.File(clim_result_path + f"{basename}_clim_output.h5", "w") as f:
			f.create_dataset("clim_output", data=final)

		del model, training_results
		del data_dict, train_df, val_df, test_df, data_config
		del train_loader, val_loader, test_loader

	torch.cuda.empty_cache()
	import gc

	gc.collect()
