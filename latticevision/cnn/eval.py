import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from latticevision.img_augment import random_augment
from latticevision.img2img.dataset import DataConfig

import pandas as pd


def eval_model(
	model: nn.Module,
	device: torch.device,
	config: DataConfig,
	test_loader: DataLoader,
	test_df: Dataset,
	plot: bool = True,
	augmentation: bool = False,
	show: bool = True,
) -> dict:
	"""
	Evaluates the CNN model on the test set by computing a number of metrics
	for each parameter (kappa2, theta, rho) and optionally plots the predicted
	vs actual values.

	Metrics computed (for each parameter):
		- RMSE: Root Mean Squared Error
		- MAE: Mean Absolute Error
		- R2: Coefficient of Determination
		- NRMSE: Normalized Root Mean Squared Error

	Args:
		model (nn.Module): The trained CNN model.
		device (torch.device): The device to run the model on.
		config (DataConfig): Data configuration object.
		test_loader (DataLoader): DataLoader for the test set.
		test_df (Dataset): The test dataset.
		plot (bool): If True, plots the predicted vs actual values. Default is True.
		augmentation (bool): If True, applies random augmentations to the fields during evaluation. Default is False.
		show (bool): If True, shows the plots. Default is True.

	Returns:
		dict: A dictionary containing the computed metrics.
	"""
	# set to eval mode
	model.eval()

	# create dicts for predictions and true values for each param.
	preds = {"kappa2": [], "theta": [], "rho": []}
	truths = {"kappa2": [], "theta": [], "rho": []}

	with torch.no_grad():
		for fields_batch, params_batch in test_loader:
			if augmentation:
				fields_batch = random_augment(fields_batch)

			fields_batch = fields_batch.to(device)
			params_batch = params_batch.to(device)

			# pass through the model
			outputs = model(fields_batch)

			# seaparate params (channel dimension)
			preds["kappa2"].append(outputs[:, 0].cpu().numpy())
			preds["theta"].append(outputs[:, 1].cpu().numpy())
			preds["rho"].append(outputs[:, 2].cpu().numpy())

			truths["kappa2"].append(params_batch[:, 0].cpu().numpy())
			truths["theta"].append(params_batch[:, 1].cpu().numpy())
			truths["rho"].append(params_batch[:, 2].cpu().numpy())

	for key in preds.keys():
		preds[key] = np.concatenate(preds[key], axis=0)  # shape: (N,)
		truths[key] = np.concatenate(truths[key], axis=0)

	preds_awght = np.exp(preds["kappa2"]) + 4
	truths_awght = np.exp(truths["kappa2"]) + 4

	def compute_metrics(true: np.ndarray, pred: np.ndarray):
		eps = 1e-6
		mse = np.mean((true - pred) ** 2)
		rmse = np.sqrt(mse)
		mae = np.mean(np.abs(true - pred))
		r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - np.mean(true)) ** 2 + eps)
		nrmse = rmse / (np.max(true) - np.min(true) + eps)
		return rmse, mae, r2, nrmse

	awght_metrics = compute_metrics(truths_awght, preds_awght)  # computed for reference
	kappa2_metrics = compute_metrics(truths["kappa2"], preds["kappa2"])
	theta_metrics = compute_metrics(truths["theta"], preds["theta"])
	rho_metrics = compute_metrics(truths["rho"], preds["rho"])

	# put into dict
	metrics = {
		"kappa2": {
			"rmse": kappa2_metrics[0],
			"mae": kappa2_metrics[1],
			"r2": kappa2_metrics[2],
			"nrmse": kappa2_metrics[3],
		},
		"theta": {
			"rmse": theta_metrics[0],
			"mae": theta_metrics[1],
			"r2": theta_metrics[2],
			"nrmse": theta_metrics[3],
		},
		"rho": {
			"rmse": rho_metrics[0],
			"mae": rho_metrics[1],
			"r2": rho_metrics[2],
			"nrmse": rho_metrics[3],
		},
	}

	# predicted vs actual plots for each param
	if plot:
		fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

		# --- log(kappa2) Predicted vs. Actual (Top Left) ---
		ax = axes[0, 0]
		ax.scatter(
			truths["kappa2"], preds["kappa2"], s=6, alpha=0.4, color="mediumslateblue"
		)
		min_val = min(np.min(truths["kappa2"]), np.min(preds["kappa2"]))
		max_val = max(np.max(truths["kappa2"]), np.max(preds["kappa2"]))
		ax.plot([min_val, max_val], [min_val, max_val], "k--")
		ax.set_title(r"$\log(\kappa^2)$ Predicted vs Actual")
		ax.set_xlabel("Actual")
		ax.set_ylabel("Predicted")

		# --- awght Predicted vs. Actual (Top Right) ---
		ax = axes[0, 1]
		ax.scatter(truths_awght, preds_awght, s=6, alpha=0.4, color="orange")
		min_val = min(np.min(truths_awght), np.min(preds_awght))
		max_val = max(np.max(truths_awght), np.max(preds_awght))
		ax.plot([min_val, max_val], [min_val, max_val], "k--")
		ax.set_title("awght Predicted vs Actual")
		ax.set_xlabel("Actual")
		ax.set_ylabel("Predicted")

		# --- theta Predicted vs. Actual (Bottom Left) ---
		ax = axes[1, 0]
		ax.scatter(truths["theta"], preds["theta"], s=6, alpha=0.4, color="deeppink")
		min_val = min(np.min(truths["theta"]), np.min(preds["theta"]))
		max_val = max(np.max(truths["theta"]), np.max(preds["theta"]))
		ax.plot([min_val, max_val], [min_val, max_val], "k--")
		ax.set_title(r"$\theta$ Predicted vs Actual")
		ax.set_xlabel("Actual")
		ax.set_ylabel("Predicted")

		# --- rho Predicted vs. Actual (Bottom Right) ---
		ax = axes[1, 1]
		ax.scatter(truths["rho"], preds["rho"], s=6, alpha=0.4, color="darkturquoise")
		min_val = min(np.min(truths["rho"]), np.min(preds["rho"]))
		max_val = max(np.max(truths["rho"]), np.max(preds["rho"]))
		ax.plot([min_val, max_val], [min_val, max_val], "k--")
		ax.set_title(r"$\rho$ Predicted vs Actual")
		ax.set_xlabel("Actual")
		ax.set_ylabel("Predicted")

		plt.tight_layout()
		if show:
			plt.show()
		else:
			plt.close(fig)

	# metrics dataframe
	rows = []
	for param, metric_vals in metrics.items():
		row = {"parameter": param}
		row.update(metric_vals)
		rows.append(row)
	df_metrics = pd.DataFrame(rows).set_index("parameter")
	print(df_metrics)

	return df_metrics


def fast_cnn_field_tiler(
	model: nn.Module,
	fields: torch.Tensor,
	device: torch.device,
	patch_batch_size: int = 1000,
	padding_mode: str = "reflect",
	verbose: bool = False,
	patch_size: int = 25,
) -> torch.Tensor:
	"""
	Faster CNN tiler for local parameter estimation across a large field,
	with no_grad() and per-batch cache clearing to avoid OOM errors.
	"""
	model = model.to(device).eval()
	B, C, H, W = fields.shape
	pad = patch_size // 2

	# field operations happen on cpu
	fields = fields.cpu()
	fields_padded = F.pad(fields, (pad, pad, pad, pad), mode=padding_mode)

	# unfold into all patches (on cpu)
	patches = (
		fields_padded.unfold(2, patch_size, 1)
		.unfold(3, patch_size, 1)
		.contiguous()
		.view(B, C, H * W, patch_size, patch_size)
		.permute(0, 2, 1, 3, 4)
		.reshape(-1, C, patch_size, patch_size)
	)

	if verbose:
		print(f"Total patches: {patches.size(0)}")

	outputs = []
	total = patches.size(0)

	# loop with no_grad + cleanup each batch
	with torch.no_grad():
		for start in range(0, total, patch_batch_size):
			end = min(start + patch_batch_size, total)
			batch = patches[start:end].to(device)

			out = model(batch)
			out = out.view(out.size(0), -1).cpu()
			outputs.append(out)

			if verbose:
				print(f"  ⋅ processed patches {start}–{end}")

			# memory management
			del batch, out
			torch.cuda.empty_cache()

	# all back on cpu
	patches = None
	all_out = torch.cat(outputs, dim=0)
	all_out = all_out.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

	return all_out


def slow_cnn_field_tiler(
	model: torch.nn.Module,
	fields: torch.Tensor,
	device: torch.device,
	padding_mode: str = "reflect",
	verbose: bool = False,
	patch_size: int = 25,
) -> torch.Tensor:
	"""
	Function for using the CNN as a local estimation tool across a large spatial field.
	This function is much slower but uses less cuda memory (primarily uses the cpu).
	This function uses a double for loop to loop over each pixel (inefficient).

	Args:
		model (nn.Module): The trained CNN model.
		fields (torch.Tensor): The input fields to estimate on.
		device (torch.device): The device to run the model on.
		padding_mode (str): The padding mode for the fields. Default is 'replicate'.
		verbose (bool): If True, prints progress updates. Default is False.

	Returns:
		torch.Tensor: The resulting global parameter fields produced by local estimation.
	"""

	patch_size = patch_size
	pad = patch_size // 2

	# make sure data is correct shape
	B, C, H, W = fields.shape

	# pad all fields
	fields_padded = F.pad(fields, (pad, pad, pad, pad), mode=padding_mode)

	# outputs go here
	output_channels = 3
	outputs = torch.zeros((B, output_channels, H, W))

	model.to(device)
	model.eval()

	with torch.no_grad():
		for b in range(B):
			if verbose:
				print(f"Processing field {b + 1}/{B}...")
			# loop over each pixel
			for i in range(H):
				for j in range(W):
					# extract patch centered at that pixel
					patch = fields_padded[b, :, i : i + patch_size, j : j + patch_size]
					patch = patch.unsqueeze(0)
					patch = patch.to(device)

					# pass through the model. prediction vals corresponds to center pixel
					pred = model(patch)
					pred = pred.squeeze()

					# in case model returns a scalar when batch size is 1, ensure we have a tensor of shape (3,)
					if pred.dim() == 0:
						pred = pred.unsqueeze(0)

					# put on cpu and save
					outputs[b, :, i, j] = pred.cpu()

	# make sure all on cpu and cuda cache clear (was having memory issues earlier)
	outputs = outputs.detach().cpu()
	torch.cuda.empty_cache()
	return outputs
