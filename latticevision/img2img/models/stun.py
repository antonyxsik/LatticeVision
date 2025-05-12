import torch.nn as nn

from latticevision.img2img.base import (
	DecoderBlock,
	EncoderBlock,
	PatchEmbeddingConv,
	TransformerEncoder,
	ModelConfig,
)


class TransUNet(nn.Module):
	"""
	TransUNet model for spatial image-to-image tasks.
	The model uses convolutional layers for encoding and decoding,
	with a vision transformer in place of the bottleneck.
	Hybrid between UNet and ViT.

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
		super(TransUNet, self).__init__()
		self.config = config

		# encoder blocks
		in_ch = config.in_channels
		self.encoders = nn.ModuleList()
		for i, out_ch in enumerate(config.enc_block_channels):
			self.encoders.append(
				EncoderBlock(
					in_ch, out_ch, num_groups=config.group_norm_groups[i], config=config
				)
			)
			in_ch = out_ch

		# The "bottleneck" is a vit
		# create patch embedding for transformer with these specified patches.
		self.patch_size_h = config.patch_size_h
		self.patch_size_w = config.patch_size_w
		self.patch_embedding = PatchEmbeddingConv(
			config=config,
			patch_channels=in_ch,  # changed from embed_dim to in_ch
		)
		# transformer
		self.transformer: nn.Sequential = nn.Sequential(
			*[TransformerEncoder(config) for _ in range(config.num_layers)]
		)
		# de-embedding
		self.de_embedding = nn.Linear(
			config.embed_dim,
			config.embed_dim * config.patch_size_h * config.patch_size_w,
		)

		# decoder blocks
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

		# encoder
		for encoder in self.encoders:
			x_conv, x = encoder(x)
			skip_connections.append(x_conv)

		# x has shape: (B, last enc channel, H/16, W/16) assuming original input divisible by 16
		B, _, H_enc, W_enc = x.shape
		# make sure H_enc % num_patches_h == 0 and W_enc % num_patches_w == 0
		assert H_enc % self.patch_size_h == 0 and W_enc % self.patch_size_w == 0, (
			"Encoded feature map dimensions must be divisible by the specified number of patches."
		)

		# transformer
		# perform patching, which takes dim from last enc channel to embed dim
		x, (H_patches, W_patches) = self.patch_embedding(x)
		# num_patches = H_patches * W_patches

		# transformer layers
		x = self.transformer(x)

		# de-embedding
		x = self.de_embedding(x)

		# new reshaping, use embed dim instead of last enc channel
		x = x.view(
			B,
			H_patches,
			W_patches,
			self.config.embed_dim,
			self.patch_size_h,
			self.patch_size_w,
		)
		x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
		x = x.view(
			B,
			self.config.embed_dim,
			H_patches * self.patch_size_h,
			W_patches * self.patch_size_w,
		)

		# decoder
		for decoder in self.decoders:
			x = decoder(x, skip_connections.pop())

		# final convolution of output
		out = self.final_conv(x)
		return out
