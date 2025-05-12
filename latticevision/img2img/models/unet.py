import torch.nn as nn

from latticevision.img2img.base import DecoderBlock, EncoderBlock, ModelConfig


class UNet(nn.Module):
	"""
	U-Net model for spatial image-to-image tasks.
	The model uses convolutional layers for encoding and decoding,
	with skip connections. Bottleneck has same number of channels
	as the transformer embedding dimension.

	Args:
	    config: ModelConfig
	        Configuration object for the model, see ModelConfig for more details.

	Methods:
		forward(x: torch.Tensor):
			Forward pass through the model.
	"""

	def __init__(
		self,
		config: ModelConfig,
	):
		super(UNet, self).__init__()
		self.config = config

		in_ch = config.in_channels
		self.encoders = nn.ModuleList()
		for i, out_ch in enumerate(config.enc_block_channels):
			self.encoders.append(
				EncoderBlock(
					in_ch, out_ch, num_groups=config.group_norm_groups[i], config=config
				)
			)
			in_ch = out_ch

		self.bottleneck = nn.Sequential(
			nn.Conv2d(
				config.enc_block_channels[-1],
				config.embed_dim,
				kernel_size=config.kernel_size,
				padding=config.padding,
			),
			nn.GroupNorm(
				num_groups=(config.embed_dim // 8), num_channels=config.embed_dim
			),
			nn.GELU(),
			nn.Conv2d(
				config.embed_dim,
				config.embed_dim,
				kernel_size=config.kernel_size,
				padding=config.padding,
			),
			nn.GroupNorm(
				num_groups=(config.embed_dim // 8), num_channels=config.embed_dim
			),
			nn.GELU(),
		)

		in_ch = config.embed_dim
		dec_block_channels = config.enc_block_channels[::-1]

		self.decoders = nn.ModuleList()
		for i, out_ch in enumerate(dec_block_channels):
			self.decoders.append(
				DecoderBlock(
					in_ch,
					out_ch,
					out_ch,
					num_groups=config.group_norm_groups[-(i + 1)],
					config=config,
				)
			)
			in_ch = out_ch

		self.final_conv = nn.Conv2d(in_ch, config.out_channels, kernel_size=1)

	def forward(self, x):
		skip_connections = []

		# encoder blocks
		for encoder in self.encoders:
			x_conv, x = encoder(x)
			skip_connections.append(x_conv)

		# bottleneck (same size as transformer embedding)
		x = self.bottleneck(x)

		# decoder blocks
		for decoder in self.decoders:
			x = decoder(x, skip_connections.pop())

		# final convolution for output
		out = self.final_conv(x)

		return out
