import torch


def set_device(
	machine: str = "local", gpu: bool = True, gpu_id: str = "cuda", verbose: bool = True
) -> torch.device:
	"""
	Sets the device for computation (CPU or GPU).

	Args:
		machine: str, optional
			"local" or "remote" machine. Default is "local".
		gpu: bool, optional
			If True, uses the GPU. Default is True.
		gpu_id: str, optional
			Specifies the GPU ID. Default is "cuda". Remote needs specification.
		verbose: bool, optional
			If True, prints the device and hardware. Default is True.

	Returns:
		- torch.device
			Device for computation
	"""
	# if gpu is true, try to use the gpu
	if gpu == True:
		if machine == "local":
			if torch.cuda.is_available():
				device = torch.device("cuda")
			elif torch.backends.mps.is_available():
				device = torch.device("mps")
			else:
				device = torch.device("cpu")

		elif machine == "remote":
			# remote requires specification
			device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")

		else:
			print("Invalid Machine, please choose either local or remote")

	# if gpu is false, use the cpu
	else:
		device = torch.device("cpu")

	# print the device and processing unit that will be used
	if verbose:
		if gpu == True:
			print(
				"Using device:", device, "\nHardware: ", torch.cuda.get_device_name(0)
			)
		else:
			print("Using device: ", device)

	return device
