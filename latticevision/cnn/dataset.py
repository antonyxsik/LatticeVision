import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from dataclasses import dataclass
import sys


@dataclass(kw_only=True)
class DataConfig:
	"""
	Config object for local CNN dataset.

	Attributes:
		file_path: str
			Path to the HDF5 file containing the data.
		sidelen: int
			Length of one side of the square field images.
		n_replicates: int
			Number of field replicates per simulation.
		n_params: int
			Number of parameters in the dataset.
		log_kappa2: bool
			If True, log-transforms the first parameter (kappa2). Default is True.
		val_size: float
			Proportion of the full dataset to include in the validation set. Default is 0.2.
		test_size: float
			Proportion of the validation dataset to make into the test set. Default is 0.2.
		random_state: int
			Random seed for reproducibility. Default is 777.
		verbose: bool
			If True, prints additional information during processing. Default is True.
	"""

	file_path: str
	fullsidelen: int
	sidelen: int
	n_replicates: int
	n_params: int
	log_kappa2: bool = True
	val_size: float = 0.2
	test_size: float = 0.2
	random_state: int = 777
	verbose: bool = True


class CNNDataClass(Dataset):
	"""
	PyTorch Dataset class for local CNN data.
	Attributes:
	    fields: np.ndarray
	        Stores the input field data.
	    params: np.ndarray
	        Stores the parameter data.

	Methods:
	    __len__: Returns the number of samples in the dataset.
	    __getitem__: Retrieves a sample from the dataset by index, adding a channel dimension for fields.
	"""

	def __init__(self, fields, params):
		self.fields = fields
		self.params = params

	def __len__(self):
		return len(self.fields)

	def __getitem__(self, idx):
		# adding channel dimension for fields
		field = torch.tensor(self.fields[idx], dtype=torch.float32)
		params = torch.tensor(self.params[idx], dtype=torch.float32)
		return field, params


def make_dataset(config: DataConfig) -> dict:
	"""
	Creates local CNN train, validation, and test datasets.

	Args:
		config: DataConfig
			Configuration object containing dataset parameters.

	Returns:
		A dict containing:
		- train_df: CNNDataClass for training data
		- val_df: CNNDataClass for validation data
		- test_df: CNNDataClass for test data
	"""
	with h5py.File(config.file_path, "r") as f:
		if config.verbose:
			print("Components in the file:", list(f.keys()))
		# extract fields
		fields = f["fields"][:]

		if config.verbose:
			print("Dataset size (MB): ", sys.getsizeof(fields) / 1024**2)
			print("Dataset size (GB): ", sys.getsizeof(fields) / 1024**3)

	# extract params
	params = fields[:, -1, : config.n_params]
	fields = fields[:, :-1, :]
	fields = fields[:, : config.n_replicates, :]

	# reshape into proper form and then extract params
	n_sims, n_fields, _ = fields.shape
	fields = fields.reshape(n_sims, n_fields, config.fullsidelen, config.fullsidelen)

	if config.sidelen != config.fullsidelen:
		fields = fields[:, :, : config.sidelen, : config.sidelen]

	# kappas make sense on a log scale
	if config.log_kappa2:
		params[:, 0] = np.log(params[:, 0])

	if config.verbose:
		print("Dims of 'fields':", fields.shape)
		print("Dims of 'params':", params.shape)

	# split into train, validation, and test
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

	# create dataset objects
	train_df = CNNDataClass(fields_train, params_train)
	val_df = CNNDataClass(fields_val, params_val)
	test_df = CNNDataClass(fields_test, params_test)

	return {"train_df": train_df, "val_df": val_df, "test_df": test_df}
