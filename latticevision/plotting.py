import random
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from latticevision.img2img.dataset import no_transform, polar_transform


def plot_example_field(
	dataset,
	config,
	idx: int,
	model_type: str = "STUN",
	field_color: str = "turbo",
	param1_color: str = "viridis",
	param2_color: str = "viridis",
	param3_color: str = "viridis",
	show: bool = True,
) -> None:
	"""
	Plots an example field at a specified index from a dataset.
	Works for both STUN and CNN models. Takes color arguments.

	Args:
	    dataset: Dataset
	        The dataset containing the fields and associated parameters.
		config:
			The configuration for the dataset.
	    idx: int
	        The index of the field to be plotted.
	    model_type: str
	        The type of model used to generate the field. Default is "STUN".
	    field_color: str
	        The color map to use for the field. Default is "turbo".
	    param1_color: str
	        The color map to use for the first parameter. Default is "viridis".
	    param2_color: str
	        The color map to use for the second parameter. Default is "viridis".
	    param3_color: str
	        The color map to use for the third parameter. Default is "viridis".
	    show: bool
	        Whether to display the plot. If False, the figure is created but not shown.

	Returns:
	    None
	"""

	if model_type == "CNN":
		# extract field and params
		field, params = dataset[idx]
		# separate params for plotting
		kappa2 = params[0]
		theta = params[1]
		rho = params[2]

		fig, ax = plt.subplots(figsize=(6, 5))
		im = ax.imshow(field[0], cmap=field_color)
		fig.colorbar(im, ax=ax, orientation="vertical")

		ax.set_title(f"Field #{idx + 1}, Replicate #1")
		ax.set_xlabel(
			rf"$\kappa^2$ = {kappa2:.4f}, $\theta$ = {theta:.2f}, $\rho$ = {rho:.2f}"
		)
		ax.invert_yaxis()

	elif model_type == "STUN":
		# extract field and params
		field = dataset[idx][0][0]
		params = dataset[idx][1]
		# kappa2 transformed back to original scale
		kappa2 = torch.exp(params[0])
		theta = params[1]
		rho = params[2]

		fig, axs = plt.subplots(1, 4, figsize=(20, 5))

		# spatial field plot
		im1 = axs[0].imshow(field, cmap=field_color)
		axs[0].set_title(f"Field at Index {idx}")
		fig.colorbar(im1, ax=axs[0], orientation="horizontal", shrink=0.8)
		axs[0].invert_yaxis()

		# kappa2 plot
		im2 = axs[1].imshow(kappa2, cmap=param1_color)
		axs[1].set_title(r"$\kappa^2$ Field")
		fig.colorbar(im2, ax=axs[1], orientation="horizontal", shrink=0.8)
		axs[1].invert_yaxis()

		# theta plot
		im3 = axs[2].imshow(theta, cmap=param2_color)
		if config.transform_function == no_transform:
			axs[2].set_title(r"$\theta$ Field")
		elif config.transform_function == polar_transform:
			axs[2].set_title(r"$f_1$ (Polar) Field")
		fig.colorbar(im3, ax=axs[2], orientation="horizontal", shrink=0.8)
		axs[2].invert_yaxis()

		# rho_x plot
		im4 = axs[3].imshow(rho, cmap=param3_color)
		if config.transform_function == no_transform:
			axs[3].set_title(r"$\rho$ Field")
		elif config.transform_function == polar_transform:
			axs[3].set_title(r"$f_2$ (Polar) Field")
		fig.colorbar(im4, ax=axs[3], orientation="horizontal", shrink=0.8)
		axs[3].invert_yaxis()

	else:
		print(
			"Invalid model type. Unable to plot field. Please use either STUN or CNN."
		)
		return

	# option set for testing
	if show:
		plt.show()
	else:
		plt.close(fig)


def plot_losses(
	train_losses: list, val_losses: list, base_losses: list = None, show: bool = True
) -> None:
	"""
	Plots the training and validation loss curves. Optionally plots the baseline loss
	if provided.

	Args:
	    train_losses: list
	        List of training loss values for each epoch.
	    val_losses: list
	        List of validation loss values for each epoch.
	    base_losses: list, optional
	        List of baseline loss values for each epoch. Default is None.
	    show: bool
	        Whether to display the plot or not.

	Returns:
		None
	"""
	fig, ax = plt.subplots(figsize=(10, 5))

	ax.plot(train_losses, label="Training Loss", color="mediumturquoise")
	ax.plot(val_losses, label="Validation Loss", color="coral")

	# add if baseline (mean) loss provided
	if base_losses is not None:
		ax.plot(base_losses, label="Baseline Loss", color="yellowgreen")

	ax.set_xlabel("Epoch")
	ax.set_ylabel("Loss")
	ax.set_title("Loss Curves")
	ax.legend()

	if show:
		plt.show()
	else:
		plt.close(fig)


def plot_img2img_samples(
	model: nn.Module,
	config,
	device: torch.device,
	test_df: Dataset,
	indices: Optional[List[int]] = None,
	random_selection: bool = True,
	num_rand_samples: int = 3,
	show: bool = True,
	awght_not_kappa2: bool = False,
	cnn_mode: bool = False,
	cnn_results: Optional[torch.Tensor] = None,
) -> None:
	"""
	Predicts and plots the true and predicted parameter fields for a few samples.
	Typically, just uses STUN model to evaluate the fields at the indices provided
	on the fly. To make life easy, a `cnn_mode` has been integrated to plot the
	results from local CNN estimation also. In order to keep things simple with
	the tiling/local estimation functions, the CNN results need to provided
	ahead of time, rather than being computed inside the function.

	Args:
	    model: nn.Module
	        The trained STUN model used to generate predictions for the fields.
		config:
			The configuration for the dataset.
	    device: torch.device
	        The device on which computations are performed.
	    test_df: Dataset
	        The dataset containing test fields and parameters.
	    indices: Optional[List[int]], optional
	        Specific indices to plot. Ignored if `random_selection` is True.
	    random_selection: bool, optional
	        If True, selects random indices for plotting. Default is True.
	    num_rand_samples: int, optional
	        Number of random samples to plot if `random_selection` is True. Default is 3.
	    show: bool, optional
	        Whether to display the plot. Default is True.
	    awght_not_kappa2: bool, optional
	        If True, plots the `awght` field instead of `kappa2`. Default is False.
		cnn_mode: bool, optional
			Whether to use CNN data. Default is False.
		cnn_results: Optional[torch.Tensor], optional
			The CNN results to use for plotting. Default is None.

	Returns:
	    None
	"""

	# in case random indices are desired
	if random_selection:
		num_samples = num_rand_samples
		indices = random.sample(range(len(test_df)), num_samples)
	else:
		num_samples = len(indices)

	fields_sample = []
	params_sample = []
	# if we are operating with cnn data, results are already provided
	if cnn_mode:
		preds_sample = []

	for i in indices:
		field, params = test_df[i]
		fields_sample.append(field)
		params_sample.append(params)
		if cnn_mode:
			preds_sample.append(cnn_results[i])

	fields_torch = torch.stack(fields_sample).to(device)
	params_torch = torch.stack(params_sample).to(device)
	if cnn_mode:
		preds_torch = torch.stack(preds_sample).to(device)
	# if working with STUN, results are created using the model
	if cnn_mode == False:
		preds_torch = model(fields_torch)

	fig, ax = plt.subplots(num_samples, 7, figsize=(26, 2.4 * num_samples))

	if num_samples == 1:
		# in case of single sample
		ax = ax.reshape(1, 7)

	for i in range(num_samples):
		# fields
		field = fields_torch[i, 0].detach().cpu().numpy()

		# true params
		kappa2 = params_torch[i, 0].detach().cpu().numpy()
		theta = params_torch[i, 1].detach().cpu().numpy()
		rho = params_torch[i, 2].detach().cpu().numpy()
		awght = np.exp(kappa2) + 4

		# predicted params
		kappa2_pred = preds_torch[i, 0].detach().cpu().numpy()
		theta_pred = preds_torch[i, 1].detach().cpu().numpy()
		rho_pred = preds_torch[i, 2].detach().cpu().numpy()
		awght_pred = np.exp(kappa2_pred) + 4

		# Plot 1: spatial field
		im0 = ax[i, 0].imshow(field, cmap="turbo")
		ax[i, 0].set_title(f"Field {indices[i] + 1}")
		fig.colorbar(im0, ax=ax[i, 0], fraction=0.046, pad=0.04)
		ax[i, 0].invert_yaxis()

		# Plot 1 and 2: true and predicted awght or kappa2
		if awght_not_kappa2:
			# true and predicted awght
			im1 = ax[i, 1].imshow(awght, cmap="viridis")
			ax[i, 1].set_title("True Awght")
			fig.colorbar(im1, ax=ax[i, 1], fraction=0.046, pad=0.04)
			ax[i, 1].invert_yaxis()

			im2 = ax[i, 2].imshow(awght_pred, cmap="viridis")
			ax[i, 2].set_title("Predicted Awght")
			fig.colorbar(im2, ax=ax[i, 2], fraction=0.046, pad=0.04)
			ax[i, 2].invert_yaxis()
		else:
			# kappa2
			im1 = ax[i, 1].imshow(kappa2, cmap="viridis")
			ax[i, 1].set_title(r"True $log(\kappa^2)$")
			fig.colorbar(im1, ax=ax[i, 1], fraction=0.046, pad=0.04)
			ax[i, 1].invert_yaxis()

			im2 = ax[i, 2].imshow(kappa2_pred, cmap="viridis")
			ax[i, 2].set_title(r"Predicted $log(\kappa^2)$")
			fig.colorbar(im2, ax=ax[i, 2], fraction=0.046, pad=0.04)
			ax[i, 2].invert_yaxis()

		# Plot 3 and 4: true and predicted theta
		if config.transform_function == no_transform:
			ax[i, 3].set_title(r"True $\theta$")
			ax[i, 4].set_title(r"Predicted $\theta$")
		elif config.transform_function == polar_transform:
			ax[i, 3].set_title(r"True $f_1$")
			ax[i, 4].set_title(r"Predicted $f_1$")

		im3 = ax[i, 3].imshow(theta, cmap="viridis")
		fig.colorbar(im3, ax=ax[i, 3], fraction=0.046, pad=0.04)
		ax[i, 3].invert_yaxis()

		im4 = ax[i, 4].imshow(theta_pred, cmap="viridis")
		fig.colorbar(im4, ax=ax[i, 4], fraction=0.046, pad=0.04)
		ax[i, 4].invert_yaxis()

		# Plot 5 and 6: true and predicted rho
		if config.transform_function == no_transform:
			ax[i, 5].set_title(r"True $\rho$")
			ax[i, 6].set_title(r"Predicted $\rho$")
		elif config.transform_function == polar_transform:
			ax[i, 5].set_title(r"True $f_2$")
			ax[i, 6].set_title(r"Predicted $f_2$")

		im5 = ax[i, 5].imshow(rho, cmap="viridis")
		fig.colorbar(im5, ax=ax[i, 5], fraction=0.046, pad=0.04)
		ax[i, 5].invert_yaxis()

		im6 = ax[i, 6].imshow(rho_pred, cmap="viridis")
		fig.colorbar(im6, ax=ax[i, 6], fraction=0.046, pad=0.04)
		ax[i, 6].invert_yaxis()

	plt.tight_layout()
	if show:
		plt.show()
	else:
		plt.close(fig)
