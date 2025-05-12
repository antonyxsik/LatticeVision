import os
import random
import numpy as np
import torch


def set_all_random_seeds(seed: int, cudnn_deterministic: bool = False):
	# python
	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)

	# numpy
	np.random.seed(seed)

	# torch
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	# cuDNN is optional (deterministic is slow)
	if cudnn_deterministic:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	else:
		# cuDNN picks fastest kernels (non‑deterministic)
		torch.backends.cudnn.deterministic = False
		torch.backends.cudnn.benchmark = True
