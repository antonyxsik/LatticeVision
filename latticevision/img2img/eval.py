import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from typing import Optional


# from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import (
	structural_similarity as ssim_metric,
	peak_signal_noise_ratio as psnr_metric,
)

from latticevision.img_augment import random_augment_clim
from latticevision.img2img.dataset import DataConfig, no_transform, polar_transform


def eval_model(
	model: torch.nn.Module,
	device: torch.device,
	config: DataConfig,
	test_loader: torch.utils.data.DataLoader,
	test_df: torch.utils.data.Dataset,
	plot: bool = True,
	augmentation: bool = False,
	show: bool = True,
	n_pixels: int = 5000,
	invert_transform: bool = False,
	cnn_mode: bool = False,
	cnn_results: Optional[torch.tensor] = None,
) -> dict:
	"""
	Evaluates the STUN model on the test set by computing a number of metrics
	for each parameter (kappa2, theta, rho) and optionally plots the predicted
	vs actual values (pixelwise). To make life easy, a `cnn_mode` has been
	integrated to plot the results from local CNN estimation also. In order to
	keep things simple with the tiling/local estimation functions, the CNN results
	need to provided ahead of time, rather than being computed inside the function.

	Metrics computed (for each parameter):
	    - RMSE (Root Mean Squared Error)
	    - MAE (Mean Absolute Error)
	    - R² (Coefficient of Determination)
	    - SSIM (Structural Similarity Index)
	    - PSNR (Peak Signal-to-Noise Ratio)
	    - NRMSE (Normalized RMSE)

	Args:
	    model (nn.Module): The trained model.
		config (DataConfig): Configuration for the dataset.
	    device (torch.device): Device to perform computations on.
	    test_loader (DataLoader): DataLoader for the test set.
	    test_df (Dataset): Test dataset (used only for computing the number of test samples).
	    plot (bool, optional): If True, plots predicted vs actual graphs. Defaults to True.
	    augmentation (bool, optional): If True, applies augmentation. Defaults to False.
		show (bool, optional): If True, shows the plot. Defaults to True.
		n_pixels (int, optional): Number of pixels to sample for the predicted vs actual plot. Defaults to 5000.
		invert_transform (bool, optional): If True, applies the inverse transform to the predicted values. Defaults to False.
		cnn_mode (bool, optional): If True, uses CNN data. Defaults to False.
		cnn_results (Optional[torch.tensor], optional): The CNN results to use for plotting. Defaults to None.

	Returns:
	    dict: Dictionary containing computed metrics for each parameter.
	"""

	model.eval()

	# create dicts for predictions and true values for each param.
	preds = {"kappa2": [], "theta": [], "rho": []}
	truths = {"kappa2": [], "theta": [], "rho": []}

	if cnn_mode == False:
		with torch.no_grad():
			for fields_batch, params_batch in test_loader:
				if augmentation:
					fields_batch, params_batch = random_augment_clim(
						fields_batch, params_batch
					)

				fields_batch = fields_batch.to(device)
				params_batch = params_batch.to(device)

				# pass through model
				outputs = model(fields_batch)

				# seaparate params (channel dimension)
				preds["kappa2"].append(outputs[:, 0, :, :].cpu().numpy())
				preds["theta"].append(outputs[:, 1, :, :].cpu().numpy())
				preds["rho"].append(outputs[:, 2, :, :].cpu().numpy())

				truths["kappa2"].append(params_batch[:, 0, :, :].cpu().numpy())
				truths["theta"].append(params_batch[:, 1, :, :].cpu().numpy())
				truths["rho"].append(params_batch[:, 2, :, :].cpu().numpy())
	else:
		outputs = cnn_results
		params_batch = test_df[:][1]
		preds["kappa2"].append(outputs[:, 0, :, :].cpu().numpy())
		preds["theta"].append(outputs[:, 1, :, :].cpu().numpy())
		preds["rho"].append(outputs[:, 2, :, :].cpu().numpy())

		truths["kappa2"].append(params_batch[:, 0, :, :].cpu().numpy())
		truths["theta"].append(params_batch[:, 1, :, :].cpu().numpy())
		truths["rho"].append(params_batch[:, 2, :, :].cpu().numpy())

	# concatenate predictions/ground-truth along the batch dimension
	for key in preds.keys():
		preds[key] = np.concatenate(preds[key], axis=0)  # shape: (N, 192, 288)
		truths[key] = np.concatenate(truths[key], axis=0)

	# apply transform to recover awght param (more familiar for LK)
	preds_awght = np.exp(preds["kappa2"]) + 4
	truths_awght = np.exp(truths["kappa2"]) + 4
	if invert_transform:
		if config.transform_function == polar_transform:
			theta = preds["theta"].copy()
			rho = preds["rho"].copy()

			preds["theta"] = np.arctan(theta / rho) / 2
			preds["rho"] = np.square(theta) + np.square(rho) + 1

	# helper function to compute metrics
	def compute_metrics(true: np.ndarray, pred: np.ndarray):
		"""
		Computes RMSE, MAE, R2, SSIM, PSNR, and NRMSE between two arrays.
		Assumes true and pred have shape (num_images, height, width).
		"""
		# eps = 1e-6
		mse = np.mean((true - pred) ** 2)
		rmse = np.sqrt(mse)
		mae = np.mean(np.abs(true - pred))
		r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - np.mean(true)) ** 2)
		nrmse = rmse / (np.max(true) - np.min(true))

		# compute SSIM and PSNR for each image and then average
		ssim_list = []
		psnr_list = []
		for i in range(true.shape[0]):
			# define data range for each image (avoid division by zero)
			data_range = np.max(true[i]) - np.min(true[i])
			if data_range == 0:
				data_range = 1.0
			ssim_val = ssim_metric(true[i], pred[i], data_range=data_range)
			psnr_val = psnr_metric(true[i], pred[i], data_range=data_range)
			ssim_list.append(ssim_val)
			psnr_list.append(psnr_val)
		ssim_avg = np.mean(ssim_list)
		psnr_avg = np.mean(psnr_list)

		return rmse, mae, r2, ssim_avg, psnr_avg, nrmse

	# compute metrics for each param
	# we dont use this variable yet, going to compute it though for tests
	awght_metrics = compute_metrics(truths_awght, preds_awght)  # noqa: F841
	kappa2_metrics = compute_metrics(truths["kappa2"], preds["kappa2"])
	theta_metrics = compute_metrics(truths["theta"], preds["theta"])
	rho_metrics = compute_metrics(truths["rho"], preds["rho"])

	# put metrics into dictionary
	metrics = {
		"kappa2": {
			"rmse": kappa2_metrics[0],
			"mae": kappa2_metrics[1],
			"r2": kappa2_metrics[2],
			"ssim": kappa2_metrics[3],
			"psnr": kappa2_metrics[4],
			"nrmse": kappa2_metrics[5],
		},
		"theta": {
			"rmse": theta_metrics[0],
			"mae": theta_metrics[1],
			"r2": theta_metrics[2],
			"ssim": theta_metrics[3],
			"psnr": theta_metrics[4],
			"nrmse": theta_metrics[5],
		},
		"rho": {
			"rmse": rho_metrics[0],
			"mae": rho_metrics[1],
			"r2": rho_metrics[2],
			"ssim": rho_metrics[3],
			"psnr": rho_metrics[4],
			"nrmse": rho_metrics[5],
		},
	}

	# plotting (optional)
	if plot:
		fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

		# --- Kappa2 predicted vs. actual plot ---
		ax = axes[0, 0]
		true_vals = truths["kappa2"].flatten()
		pred_vals = preds["kappa2"].flatten()
		idx = np.random.choice(
			len(true_vals), size=min(n_pixels, len(true_vals)), replace=False
		)
		ax.scatter(
			true_vals[idx], pred_vals[idx], s=6, alpha=0.2, color="mediumslateblue"
		)
		min_val = min(true_vals.min(), pred_vals.min())
		max_val = max(true_vals.max(), pred_vals.max())
		ax.plot([min_val, max_val], [min_val, max_val], "k--")
		ax.set_title(r"$log(\kappa^2)$ Predicted vs Actual")
		ax.set_xlabel("Actual")
		ax.set_ylabel("Predicted")

		# --- Awght predicted vs. actual plot ---
		ax = axes[0, 1]
		true_vals = truths_awght.flatten()
		pred_vals = preds_awght.flatten()
		idx = np.random.choice(
			len(true_vals), size=min(n_pixels, len(true_vals)), replace=False
		)
		ax.scatter(true_vals[idx], pred_vals[idx], s=6, alpha=0.2, color="orange")
		min_val = min(true_vals.min(), pred_vals.min())
		max_val = max(true_vals.max(), pred_vals.max())
		ax.plot([min_val, max_val], [min_val, max_val], "k--")
		ax.set_title("awght Predicted vs Actual")
		ax.set_xlabel("Actual")
		ax.set_ylabel("Predicted")

		# --- theta predicted vs. actual plot ---
		ax = axes[1, 0]
		true_vals = truths["theta"].flatten()
		pred_vals = preds["theta"].flatten()
		idx = np.random.choice(
			len(true_vals), size=min(n_pixels, len(true_vals)), replace=False
		)
		ax.scatter(true_vals[idx], pred_vals[idx], s=6, alpha=0.2, color="deeppink")
		min_val = min(true_vals.min(), pred_vals.min())
		max_val = max(true_vals.max(), pred_vals.max())
		ax.plot([min_val, max_val], [min_val, max_val], "k--")
		if config.transform_function == no_transform:
			ax.set_title(r"$\theta$ Predicted vs Actual")
		elif config.transform_function == polar_transform:
			ax.set_title(r"$f_1$ Predicted vs Actual")
			if invert_transform:
				ax.set_title(r"$\theta$ Predicted vs Actual")
		ax.set_xlabel("Actual")
		ax.set_ylabel("Predicted")

		# --- rho predicted vs. actual plot ---
		ax = axes[1, 1]
		true_vals = truths["rho"].flatten()
		pred_vals = preds["rho"].flatten()
		idx = np.random.choice(
			len(true_vals), size=min(n_pixels, len(true_vals)), replace=False
		)
		ax.scatter(
			true_vals[idx], pred_vals[idx], s=6, alpha=0.2, color="darkturquoise"
		)
		min_val = min(true_vals.min(), pred_vals.min())
		max_val = max(true_vals.max(), pred_vals.max())
		ax.plot([min_val, max_val], [min_val, max_val], "k--")
		if config.transform_function == no_transform:
			ax.set_title(r"$\rho$ Predicted vs Actual")
		elif config.transform_function == polar_transform:
			ax.set_title(r"$f_2$ Predicted vs Actual")
			if invert_transform:
				ax.set_title(r"$\rho$ Predicted vs Actual")
		ax.set_xlabel("Actual")
		ax.set_ylabel("Predicted")

		plt.tight_layout()

		if show:
			plt.show()
		else:
			plt.close(fig)

	rows = []
	for param, metric_vals in metrics.items():
		row = {"parameter": param}
		row.update(metric_vals)
		rows.append(row)
	df_metrics = pd.DataFrame(rows).set_index("parameter")

	# print the dataframe
	print(df_metrics)

	return df_metrics

	# print the metrics regardless of plot or not
	# print("Evaluation Metrics:")
	# for param, m in metrics.items():
	# 	print(f"{param}:")
	# 	for met_name, value in m.items():
	# 		print(f"  {met_name}: {value:.4f}")

	# return metrics
