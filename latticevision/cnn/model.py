import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, List, Union


@dataclass(kw_only=True)
class ModelConfig:
	"""
	Model configuration for local estimation tasks with CNN.

	Attributes:
	    in_channels (int): Number of input channels.
	    out_params (int): Number of output parameters.
	    conv_channels (List[int]): Number of output channels for each convolutional layer.
	    kernel_size (Tuple[int, int]): Size of the convolutional kernel.
	    stride (Tuple[int, int]): Stride of the convolutional kernel.
	    pool_kernel_size (Tuple[int, int]): Size of the pooling kernel.
	    pool_stride (Tuple[int, int]): Stride of the pooling kernel.
	    linear_sizes (List[int]): Number of neurons for each linear layer.
	"""

	sidelen: int = 25
	in_channels: int = 30
	out_params: int = 3
	conv_channels: List[int] = (64, 128, 256)
	# kernel_size: Tuple[int, int] = (3, 3)
	kernel_sizes: List[int] = (10, 7, 5)
	stride: Tuple[int, int] = (1, 1)
	pool_kernel_size: Tuple[int, int] = (2, 2)
	pool_stride: Tuple[int, int] = (2, 2)
	linear_sizes: List[int] = (256, 128, 64)
	padding: Union[int, str] = 0


class CNN(nn.Module):
	"""
	CNN model for local estimation tasks.
	Attributes:
	    config (ModelConfig): Configuration for the model.

	Methods:
		forward(x: torch.Tensor):
			Forward pass through the model.
	"""

	def __init__(
		self,
		config: ModelConfig,
	):
		super(CNN, self).__init__()
		self.config = config

		conv_layers = []
		in_ch = config.in_channels

		for out_ch, ksz in zip(config.conv_channels, config.kernel_sizes):
			conv_layers.append(
				nn.Conv2d(
					in_channels=in_ch,
					out_channels=out_ch,
					kernel_size=ksz,
					stride=config.stride,
					padding=config.padding,
				)
			)
			# using GELU activation
			conv_layers.append(nn.GELU())
			in_ch = out_ch

		# final conv list is sequential
		self.conv_layers = nn.Sequential(*conv_layers)

		# max pooling layers to downsample and focus on broader features
		self.pool = nn.MaxPool2d(
			kernel_size=config.pool_kernel_size,
			stride=config.pool_kernel_size,
		)

		# output size for flattening
		in_size = in_ch * 3 * 3

		# create fully connected layers
		fc_layers = []
		for size in config.linear_sizes:
			fc_layers.append(nn.Linear(in_size, size))
			fc_layers.append(nn.GELU())
			in_size = size

		# additional layer required to output the correct number of parameters
		fc_layers.append(nn.Linear(in_size, config.out_params))

		# final fc list is sequential
		self.fc_layers = nn.Sequential(*fc_layers)

	def forward(self, x):
		# convolutional layers
		x = self.conv_layers(x)
		# pooling (downsample)
		x = self.pool(x)
		# reshape (flatten) for fc layers
		x = x.view(x.size(0), -1)
		# fully connected layers
		x = self.fc_layers(x)

		return x
