import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass

# from torchvision import models
from torch.utils.data import DataLoader, Dataset

from latticevision.img_augment import random_augment_clim


@dataclass(kw_only=True)
class TrainingConfig:
	"""
	Config object for training img2img models.

	Attributes:
	    model: nn.Module
	        The neural network model to be trained (e.g., STUN or CNN).
	    device: torch.device
	        The device on which computations are performed (e.g., "cuda" or "cpu").
	    train_loader: DataLoader
	        DataLoader for training data.
	    val_loader: DataLoader
	        DataLoader for validation data.
	    train_df: Dataset
	        The training dataset, used to calculate the training loss.
	    val_df: Dataset
	        The validation dataset, used to calculate the validation loss.
	    lr: float, optional
	        The learning rate for the optimizer. Default is 5e-5.
	    n_epochs: int, optional
	        Maximum number of epochs for training. Default is 100.
	    stop_patience: int, optional
	        Number of epochs with no improvement to trigger early stopping. Default is 10.
	    scheduler_patience: int, optional
	        Number of epochs with no improvement to reduce learning rate. Default is 5.
	    scheduler_factor: float, optional
	        Factor to reduce learning rate when validation loss plateaus. Default is 0.5.
	    augmentation: bool, optional
	        If True, applies random augmentations to the both spatial and param fields. Default is False.
	    save: bool, optional
	        If True, saves the best model weights. Default is True.
		save_directory: str, optional
			Directory to save the model weights. Default is "../results/model_wghts/".
		savename: str, optional
			Name of the file to save the model weights. Default is "stun_wghts.pth".
	    verbose: bool, optional
	        If True, prints progress, including loss values and learning rate updates. Default is True.
		normalize: bool, optional
			If true, normalizes only for the loss computation (not the actual outputs). This regulates the
			magnitude of the params (otherwise |log(kappa2)| will dominate the loss). This is done
			in the loss computation only, so the model will still output the original values.
			Default is True.
		shuffle: bool, optional
			If true, shuffles the input fields for permutation invariance (soft constraint). Default is True.
	"""

	model: nn.Module
	device: torch.device
	train_loader: DataLoader
	val_loader: DataLoader
	train_df: Dataset
	val_df: Dataset
	lr: float = 5e-5
	n_epochs: int = 200
	stop_patience: int = 10
	scheduler_patience: int = 5
	scheduler_factor: float = 0.5
	augmentation: bool = False
	save: bool = True
	save_directory: str = "../results/model_wghts/"
	savename: str = "stun_wghts.pth"
	verbose: bool = True
	normalize: bool = True
	shuffle: bool = True


def train_model(config: TrainingConfig) -> dict:
	"""
	Trains a STUN model with early stopping and learning rate scheduling, returning the best model and loss histories.

	Args:
	    config (TrainingConfig):
			- Configuration object containing training parameters and settings.

	Returns:
	    dict: A dict containing:
	        - model: nn.Module
	            The trained model with the best validation weights loaded.
	        - train_losses: list
	            List of training loss values over epochs.
			- baseline_losses: list
				List of baseline loss values over epochs.
	        - val_losses: list
	            List of validation loss values over epochs.
	"""
	model = config.model

	if config.verbose:
		print(f"Training for {config.n_epochs} epochs with learning rate {config.lr}.")
		if config.save == True:
			print(
				"Model weights will be saved at "
				+ config.save_directory
				+ config.savename
			)
		if config.save == False:
			print("Model weights will not be saved.")
		if config.augmentation:
			print("Augmentation has been enabled.")
		if config.augmentation == False:
			print("Augmentation has been disabled.")
		if config.normalize:
			print("Normalization has been enabled.")
		if config.normalize == False:
			print("Normalization has been disabled.")
		if config.shuffle:
			print("Shuffling has been enabled.")
		if config.shuffle == False:
			print("Shuffling has been disabled.")

	# calculate mean and std across entire training set for normalization
	if config.normalize:
		kappa2_mean = torch.mean(config.train_df[:][1][:, 0, :, :])
		kappa2_std = torch.std(config.train_df[:][1][:, 0, :, :])

		theta_mean = torch.mean(config.train_df[:][1][:, 1, :, :])
		theta_std = torch.std(config.train_df[:][1][:, 1, :, :])

		rho_mean = torch.mean(config.train_df[:][1][:, 2, :, :])
		rho_std = torch.std(config.train_df[:][1][:, 2, :, :])

	# to avoid division by zero
	eps = 1e-6

	# hyperparameter settings
	criterion = nn.MSELoss()
	optimizer = optim.AdamW(model.parameters(), lr=config.lr)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode="min",
		factor=config.scheduler_factor,
		patience=config.scheduler_patience,
		# verbose=True,
	)

	patience = config.stop_patience
	best_val_loss = float("inf")
	early_stop_counter = 0

	# items to return
	train_losses = []
	val_losses = []
	# baseline loss sees if the model learns something better
	# than just predicting the mean of the training data
	baseline_losses = []
	best_model_wts = model.state_dict()

	# training
	for epoch in range(config.n_epochs):
		model.train()
		train_loss_current = 0
		baseline_losses_current = 0

		# TRAINING LOOP ---------------------------------------------------------
		for batch in config.train_loader:
			fields_batch, params_batch = batch

			# augmentation happens half the time
			if config.augmentation:
				aug_choice = np.random.choice([True, False])
				if aug_choice:
					fields_batch, params_batch = random_augment_clim(
						fields_batch, params_batch
					)

			fields_batch, params_batch = (
				fields_batch.to(config.device),
				params_batch.to(config.device),
			)

			# shufflying for permutation invariance (soft constraint)
			if config.shuffle:
				fields_batch = fields_batch[
					:, torch.randperm(fields_batch.size(1)), :, :
				]

			optimizer.zero_grad()

			# print(f"{fields_batch.shape = }")
			output = model(fields_batch)

			if config.normalize:
				# normalized loss computation
				kappa2_norm = (params_batch[:, 0, :, :] - kappa2_mean) / kappa2_std
				theta_norm = (params_batch[:, 1, :, :] - theta_mean) / theta_std
				rho_norm = (params_batch[:, 2, :, :] - rho_mean) / rho_std
				params_batch_norm = torch.stack(
					[kappa2_norm, theta_norm, rho_norm], dim=1
				)

				kappa2_output_norm = (output[:, 0, :, :] - kappa2_mean) / kappa2_std
				theta_output_norm = (output[:, 1, :, :] - theta_mean) / theta_std
				rho_output_norm = (output[:, 2, :, :] - rho_mean) / rho_std
				output_norm = torch.stack(
					[kappa2_output_norm, theta_output_norm, rho_output_norm], dim=1
				)

				kappa2_loss = criterion(kappa2_output_norm, kappa2_norm)
				theta_loss = criterion(theta_output_norm, theta_norm)
				rho_loss = criterion(rho_output_norm, rho_norm)
				loss = kappa2_loss + theta_loss + rho_loss

				# compute baseline mean loss
				mean_params = torch.mean(params_batch_norm, dim=(2, 3), keepdim=True)
				mean_params = mean_params.expand_as(params_batch_norm)
				mean_loss = criterion(mean_params, params_batch_norm)

			else:
				kappa2_loss = criterion(output[:, 0, :, :], params_batch[:, 0, :, :])
				theta_loss = criterion(output[:, 1, :, :], params_batch[:, 1, :, :])
				rho_loss = criterion(output[:, 2, :, :], params_batch[:, 2, :, :])
				loss = kappa2_loss + theta_loss + rho_loss

				# compute baseline mean loss
				mean_params = torch.mean(params_batch, dim=(2, 3), keepdim=True)
				mean_params = mean_params.expand_as(params_batch)
				mean_loss = criterion(mean_params, params_batch)

			# backpropagation
			loss.backward()
			optimizer.step()
			# append to training loss
			train_loss_current += loss.item() * fields_batch.size(0)
			# append to baseline loss
			baseline_losses_current += mean_loss.item() * fields_batch.size(0)

		train_loss = train_loss_current / len(config.train_df)
		train_losses.append(train_loss)

		baseline_loss = baseline_losses_current / len(config.train_df)
		baseline_losses.append(baseline_loss)

		# VALIDATION LOOP ---------------------------------------------------------
		model.eval()
		val_loss_current = 0

		with torch.no_grad():
			for batch in config.val_loader:
				fields_batch, params_batch = batch

				if config.augmentation:
					aug_choice = np.random.choice([True, False])
					if aug_choice:
						fields_batch, params_batch = random_augment_clim(
							fields_batch, params_batch
						)

				fields_batch, params_batch = (
					fields_batch.to(config.device),
					params_batch.to(config.device),
				)

				if config.shuffle:
					fields_batch = fields_batch[
						:, torch.randperm(fields_batch.size(1)), :, :
					]

				# print(input_batch.shape)
				output = model(fields_batch)

				if config.normalize:
					# normalized loss computation
					kappa2_norm = (params_batch[:, 0, :, :] - kappa2_mean) / kappa2_std
					theta_norm = (params_batch[:, 1, :, :] - theta_mean) / theta_std
					rho_norm = (params_batch[:, 2, :, :] - rho_mean) / rho_std
					params_batch_norm = torch.stack(
						[kappa2_norm, theta_norm, rho_norm], dim=1
					)

					kappa2_output_norm = (output[:, 0, :, :] - kappa2_mean) / kappa2_std
					theta_output_norm = (output[:, 1, :, :] - theta_mean) / theta_std
					rho_output_norm = (output[:, 2, :, :] - rho_mean) / rho_std
					output_norm = torch.stack(
						[kappa2_output_norm, theta_output_norm, rho_output_norm],
						dim=1,
					)

					# loss = criterion(output_norm, params_batch_norm)
					kappa2_loss = criterion(kappa2_output_norm, kappa2_norm)
					theta_loss = criterion(theta_output_norm, theta_norm)
					rho_loss = criterion(rho_output_norm, rho_norm)
					loss = kappa2_loss + theta_loss + rho_loss
				else:
					kappa2_loss = criterion(
						output[:, 0, :, :], params_batch[:, 0, :, :]
					)
					theta_loss = criterion(output[:, 1, :, :], params_batch[:, 1, :, :])
					rho_loss = criterion(output[:, 2, :, :], params_batch[:, 2, :, :])
					loss = kappa2_loss + theta_loss + rho_loss

				# append to val loss
				val_loss_current += loss.item() * fields_batch.size(0)

		val_loss = val_loss_current / len(config.val_df)
		val_losses.append(val_loss)

		current_lr = scheduler.get_last_lr()[0]  # optimizer.param_groups[0]["lr"]
		if config.verbose:
			print(f"Epoch {epoch + 1}/{config.n_epochs}")
			print(
				f"Train Loss: {train_loss:.6f} - Base (Mean) Loss: {baseline_loss:.6f} - Val Loss: {val_loss:.6f}"
			)
			print(f"Learning Rate: {current_lr:.6f}")

		# learning rate scheduler step
		scheduler.step(val_loss)

		# check for the best validation loss and save the best model weights
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_model_wts = model.state_dict()
			early_stop_counter = 0
		else:
			early_stop_counter += 1

		# early stopping conditional
		if early_stop_counter >= patience:
			if config.verbose:
				print(
					f"Early stopping at epoch {epoch + 1}. Best validation loss: {best_val_loss:.6f}"
				)
			break

	# save different models weights in different locations based on modeltype string
	if config.save:
		torch.save(model.state_dict(), config.save_directory + config.savename)

	# load the best weights after training
	model.load_state_dict(best_model_wts)
	print("Training complete. Best model weights loaded.")

	return {
		"model": model,
		"train_losses": train_losses,
		"baseline_losses": baseline_losses,
		"val_losses": val_losses,
	}
