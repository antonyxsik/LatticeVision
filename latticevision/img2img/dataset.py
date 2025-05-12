from typing import Callable
import h5py
import numpy as np
import torch
import sys
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from dataclasses import dataclass


def no_transform(params: np.ndarray) -> np.ndarray:
	return params


# experimenting with parameter transformations
def polar_transform(params: np.ndarray) -> np.ndarray:
	"""
	Transforms the input params according to Rai and Brown (2023).
	"""
	theta = params[:, 1, :, :]
	rho = params[:, 2, :, :]
	f1 = np.sqrt(rho - 1) * np.sin(2 * theta)
	f2 = np.sqrt(rho - 1) * np.cos(2 * theta)
	params[:, 1, :, :] = f1
	params[:, 2, :, :] = f2
	return params


@dataclass(kw_only=True)
class DataConfig:
	"""
	Config object for dataset.

	Attributes:
		file_path: str
			Path to the HDF5 file containing the data.
		n_rows: int
			Number of rows in the field images.
		n_cols: int
			Number of columns in the field images.
		n_replicates: int
			Number of field replicates per simulation.
		transform_function: Callable
			Function to apply to the data. Options are currently:
			polar repameterization.
		n_params: int
			Number of parameters in the dataset.
		log_kappa2: bool
			If True, log-transforms the first parameter (kappa2). Default is True.
		shift_theta: bool
			If True, shifts the theta parameter to be between pi/2 and -pi/2. Default is True.
		val_size: float
			Proportion of the full dataset to include in the validation set. Default is 0.2.
		test_size: float
			Proportion of the validation dataset to make into the test set. Default is 0.2.
		random_state: int
			Random seed for reproducibility. Default is 777.
		verbose: bool
			If True, prints dataset information. Default is True.
	"""

	file_path: str
	n_rows: int = 192
	n_cols: int = 288
	n_replicates: int
	transform_function: Callable = no_transform
	n_params: int = 3
	log_kappa2: bool = True
	shift_theta: bool = True
	val_size: float = 0.2
	test_size: float = 0.2
	random_state: int = 777
	verbose: bool = True


class ClimDataClass(Dataset):
	"""
	Custom PyTorch dataset for handling STUN model data, including fields, parameters, and configuration.

	Attributes:
	    fields: np.ndarray
	        Stores the input field data.
	    param_fields: np.ndarray
	        Stores the parameter field data.

	Methods:
	    __len__():
	        Returns the total number of samples in the dataset.
	    __getitem__(idx: int):
	        Retrieves a sample with its replicated fields and parameters at a given index as PyTorch tensors.
	"""

	def __init__(self, fields, param_fields):
		self.fields = fields
		self.param_fields = param_fields

	def __len__(self):
		return len(self.fields)

	def __getitem__(self, idx):
		field = torch.tensor(self.fields[idx], dtype=torch.float32)
		params = torch.tensor(self.param_fields[idx], dtype=torch.float32)
		return field, params


def make_dataset(config: DataConfig) -> dict:
	"""
	Loads and preprocesses img2img data from an HDF5 file,
	reshaping and splitting it into training, validation, and test sets.

	Args:
		config: DataConfig
			Configuration object for the dataset, see DataConfig for more details.

	Returns:
		dict: A dict containing:
			train_df: Dataset
				Training dataset for the img2img model.
			val_df: Dataset
				Validation dataset for the img2img model.
			test_df: Dataset
				Test dataset for the img2img model.
	"""

	with h5py.File(config.file_path, "r") as f:
		if config.verbose:
			print("Components in the file:", list(f.keys()))
		# extract fields
		fields = f["fields"][:]

		if config.verbose:
			print("Dataset size (MB): ", sys.getsizeof(fields) / 1024**2)
			print("Dataset size (GB): ", sys.getsizeof(fields) / 1024**3)

	# reshape into proper form and then extract params
	n_sims, n_fields, _ = fields.shape
	fields = fields.reshape(n_sims, n_fields, config.n_rows, config.n_cols)
	# take the last fields as the params
	params = fields[:, -config.n_params :, :, :]
	# start at the beginning for replicates
	fields = fields[:, 0 : config.n_replicates, :, :]

	if config.verbose:
		print("Fields shape: ", fields.shape)
		print("Params shape: ", params.shape)

	# log transform kappa2
	if config.log_kappa2:
		kappa2 = params[:, 0, :, :]
		params[:, 0, :, :] = np.log(kappa2)

	if config.shift_theta:
		theta = params[:, 1, :, :]
		params[:, 1, :, :] = np.pi / 2 - theta

	# apply transform function
	params = config.transform_function(params)

	# train-val-test split
	fields_train, fields_temp, params_train, params_temp = train_test_split(
		fields, params, test_size=config.val_size, random_state=config.random_state
	)
	fields_val, fields_test, params_val, params_test = train_test_split(
		fields_temp,
		params_temp,
		test_size=config.test_size,
		random_state=config.random_state,
	)

	if config.verbose:
		print("Train fields shape: ", fields_train.shape)
		print("Train params shape: ", params_train.shape)
		print("Validation fields shape: ", fields_val.shape)
		print("Validation params shape: ", params_val.shape)
		print("Test fields shape: ", fields_test.shape)
		print("Test params shape: ", params_test.shape)

	train_df = ClimDataClass(fields_train, params_train)
	val_df = ClimDataClass(fields_val, params_val)
	test_df = ClimDataClass(fields_test, params_test)
	# del fields, params, fields_train, params_train, fields_val, params_val, fields_test, params_test

	return {"train_df": train_df, "val_df": val_df, "test_df": test_df}
