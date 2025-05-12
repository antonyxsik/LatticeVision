import torch.nn as nn

from latticevision.img2img.base import (
	PatchEmbeddingConv,
	TransformerEncoder,
	ModelConfig,
)


class ViT(nn.Module):
	"""
	A Vision Transformer that uses the specified patch sizes for splitting the input,
	then applies a series of Transformer encoder layers and reconstructs the output.

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
		super().__init__()
		# store config
		self.config: ModelConfig = config

		self.patch_embedding: PatchEmbeddingConv = PatchEmbeddingConv(
			config=config,
			patch_channels=config.in_channels,
		)

		# transformer encoder (post norms, with dropout and GELU)
		self.transformer: nn.Sequential = nn.Sequential(
			*[TransformerEncoder(config) for _ in range(config.num_layers)]
		)

		# takes a single embedding vector to a patch
		self.de_embedding: nn.Linear = nn.Linear(
			config.embed_dim,
			config.patch_size_h * config.patch_size_w * config.out_channels,
		)

	def forward(self, x):
		B, C, H, W = x.shape
		# perform patch and positional embedding
		x, (H_patches, W_patches) = self.patch_embedding(x)

		# transformer layers
		x = self.transformer(x)

		# de-embed patches back to single channel spatial dims
		x = self.de_embedding(x)

		# reshaping to correct image size
		x = x.view(
			B,
			H_patches,
			W_patches,
			self.config.patch_size_h,
			self.config.patch_size_w,
			self.config.out_channels,
		)
		x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, self.config.out_channels, H, W)
		# old reshaping (keep just in case)
		# x = x.view(B, H_patches, W_patches, self.config.patch_size_h, self.config.patch_size_w).permute(0, 3, 1, 4, 2).reshape(B, -1, H, W)
		return x
