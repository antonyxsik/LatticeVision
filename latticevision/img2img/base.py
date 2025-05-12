import math
import torch
import torch.nn as nn

from torchtune.modules import MultiHeadAttention as TT_MultiHeadAttention
from torchtune.modules import RotaryPositionalEmbeddings

from dataclasses import dataclass
from typing import Tuple, List


# abstract base class for positional embeddings
class PosEmbed(nn.Module):
	"""This class provides a structure for pos embedding classes to inherit from"""

	at_patch_embed: bool

	def __init__(self, embed_dim: int, max_n: int):
		super().__init__()
		# raise NotImplementedError(
		# 	"this is an abstract class, use something that inherits from it,
		#  like `NullPosEmbed`, `LearnedPosEmbed`, `SinusoidalPosEmbed`, or `RotaryPosEmbed`."
		# )


class NullPosEmbed(PosEmbed):
	"""Base class for testing without positional embeddings"""

	at_patch_embed: bool = True

	def __init__(self, embed_dim: int, max_n: int):
		super().__init__(embed_dim, max_n)
		self.embed_dim = embed_dim

	def forward(self, n: int) -> torch.Tensor:
		# Return a zero tensor of shape [embed_dim, n]
		return torch.zeros(self.embed_dim, n)


class LearnedPosEmbed(PosEmbed):
	"""Learnable positional embeddings"""

	at_patch_embed: bool = True

	def __init__(self, embed_dim: int, max_n: int):
		super().__init__(embed_dim, max_n)
		# essentially just an additional, learned layer
		self.pos_embed: nn.Embedding = nn.Embedding(max_n, embed_dim)

	def forward(self, n: int) -> torch.Tensor:
		# print(self.pos_embed.weight.device)
		return self.pos_embed(
			torch.arange(n, device=self.pos_embed.weight.device)
		).T  # transpose to shape (embed_dim, n)


class SinusoidalPosEmbed(PosEmbed):
	"""Sinusoidal positional embeddings"""

	at_patch_embed: bool = True

	def __init__(self, embed_dim: int, max_n: int):
		super().__init__(embed_dim, max_n)
		self.embed_dim: int = embed_dim
		self.max_n: int = max_n

	def forward(self, n: int):
		# generate the fixed positional encodings
		pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
		div_term = torch.exp(
			torch.arange(0, self.embed_dim, 2) * (-math.log(10000.0) / self.embed_dim)
		)
		pos_encoding = torch.zeros(n, self.embed_dim)
		# apply sin to even inds, cos to odd inds
		pos_encoding[:, 0::2] = torch.sin(pos * div_term)
		pos_encoding[:, 1::2] = torch.cos(pos * div_term)
		return pos_encoding.T  # transpose to shape (embed_dim, n)


class RotaryPosEmbed(PosEmbed):
	"""Rotary positional embeddings, dependent on torchtune module"""

	at_patch_embed: bool = False

	def __init__(self, embed_dim: int, max_n: int):
		super().__init__(embed_dim, max_n)
		self.rotary_pos_embed: RotaryPositionalEmbeddings = RotaryPositionalEmbeddings(
			dim=embed_dim,
			max_seq_len=max_n,
		)

	# forward method allows for more flexibility with arguments passed
	def forward(self, x: torch.Tensor, *args, **kwargs):
		return self.rotary_pos_embed(x, *args, **kwargs)


@dataclass(kw_only=True)
class ModelConfig:
	"""
	Model configuration for image-to-image tasks.

	Args:
	    in_channels: int
	        Number of input channels.
	    out_channels: int
	        Number of output channels.
	    embed_dim: int
	        Dimension of the embedding vectors.
	    enc_block_channels: List[int]
	        Number of channels in each convolutional encoder block.
	    dec_block_channels: List[int]
	        Number of channels in each convolutional decoder block.
	    group_norm_groups: List[int]
	        Number of groups for GroupNorm in each block.
	    kernel_size: Tuple[int, int]
	        Size of the convolutional kernel.
	    stride: Tuple[int, int]
	        Stride of the convolutional kernel.
	    padding: int
	        Padding for the convolutional kernel.
	    pool_kernel_size: Tuple[int, int]
	        Size of the pooling kernel.
	    pool_stride: Tuple[int, int]
	        Stride of the pooling kernel.
	    patch_size_h: int
	        Height of the patch size.
	    patch_size_w: int
	        Width of the patch size.
	    num_heads: int
	        Number of attention heads.
	    mlp_dim: int
	        Dimension of the feedforward network.
	    num_layers: int
	        Number of transformer layers.
	    dropout: float
	        Dropout rate.
	    patch_conv_bias: bool
	        Whether to include bias in the patch convolutional layer.
	    pos_embed_max_n_axis: int
	        Maximum number of patches in the x and y axes.
	    pos_embed_cls: type[PosEmbed]
	        Class for positional embeddings.
	    head_bias: bool
	        Whether to include bias in the attention heads.

		Properties:
			padding: int
				Calculates the padding for the convolutional kernel.
			patch_size_tup: tuple[int, int]
				Returns the patch size as a tuple.
			head_dim: int
				Calculates the dimension of the attention heads
	"""

	# General settings -----
	in_channels: int = 30
	out_channels: int = 3
	embed_dim: int = 768

	# Convolutional settings -----
	enc_block_channels: List[int] = (64, 128, 256, 512)
	# dec_block_channels: List[int] = (512, 256, 128, 64)block
	group_norm_groups: List[int] = (8, 16, 32, 64)
	kernel_size: Tuple[int, int] = (3, 3)
	stride: Tuple[int, int] = (1, 1)

	# for same padding use kernel_size[0]//2
	@property
	def padding(self) -> int:
		return self.kernel_size[0] // 2

	pool_kernel_size: Tuple[int, int] = (2, 2)
	pool_stride: Tuple[int, int] = (2, 2)

	# Transformer settings -----
	# set patch sizes each time for diff models
	# default is None so UNet can use the config
	patch_size_h: int = None
	patch_size_w: int = None
	num_heads: int = 12
	mlp_dim: int = 3072
	num_layers: int = 12
	dropout: float = 0.1
	patch_conv_bias: bool = False
	pos_embed_max_n_axis: int = 64
	# default to learned embeddings
	pos_embed_cls: type[PosEmbed] = LearnedPosEmbed
	head_bias: bool = False

	@property
	def patch_size_tup(self) -> tuple[int, int]:
		return (self.patch_size_h, self.patch_size_w)

	@property
	def head_dim(self) -> int:
		return self.embed_dim // self.num_heads


class EncoderBlock(nn.Module):
	"""
	Convolutional encoder block with pooling for downsampling. Used to progressively
	reduce spatial dimensions while increasing feature representation.

	Args:
	    in_channels: int
	        Number of input channels.
	    out_channels: int
	        Number of output channels.
		num_groups: int
			Number of groups for GroupNorm.

	Methods:
	    forward(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
	        Applies 2*(convolution, groupnorm, GELU, and dropout), followed by a max pooling layer.
	        Returns the transformed feature map and the pooled feature map for skip connections.
	"""

	def __init__(self, in_channels, out_channels, num_groups, config: ModelConfig):
		super(EncoderBlock, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(
				in_channels,
				out_channels,
				kernel_size=config.kernel_size,
				padding=config.padding,
			),
			# GroupNorm is not dependent on batch size, also good balance between layernorm and instancenorm
			nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
			# GELU for smoother gradients
			nn.GELU(),
			# nn.Dropout2d(p=0.1),
			nn.Conv2d(
				out_channels,
				out_channels,
				kernel_size=config.kernel_size,
				padding=config.padding,
			),
			nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
			nn.GELU(),
			# nn.Dropout2d(p=0.1),
		)
		# pooling layer halves the spatial dimensions
		self.pool = nn.MaxPool2d(
			kernel_size=config.pool_kernel_size, stride=config.pool_stride
		)

	def forward(self, x):
		x = self.conv(x)
		x_pooled = self.pool(x)
		# return both for skip connections
		return x, x_pooled


class DecoderBlock(nn.Module):
	"""
	Convolutional decoder block with transposed convolution for upsampling and skip connections.
	Used to progressively restore spatial dimensions and combine extracted spatial features.

	Args:
	    in_channels: int
	        Number of input channels.
	    skip_channels: int
	        Number of channels from skip connection.
	    out_channels: int
	        Number of output channels.
		num_groups: int
			Number of groups for GroupNorm.

	Methods:
	    forward(x: torch.Tensor, skip_x: torch.Tensor) -> torch.Tensor
	        Upsamples the input and concatenates it with the skip connection feature map, then applies
	        convolutions to refine the output. Returns the upsampled feature map.
	    Essentially symmetric to the encoder block after upsampling: 2* (convs, groupnorm, gelu, dropout).
	"""

	def __init__(
		self, in_channels, skip_channels, out_channels, num_groups, config: ModelConfig
	):
		super(DecoderBlock, self).__init__()

		# upsampling layer doubles spatial dimensions
		self.upconv = nn.ConvTranspose2d(
			in_channels,
			out_channels,
			kernel_size=config.pool_kernel_size,
			stride=config.pool_stride,
		)

		# convolutions symmetric to encoder block
		self.conv = nn.Sequential(
			nn.Conv2d(
				out_channels + skip_channels,
				out_channels,
				kernel_size=config.kernel_size,
				padding=config.padding,
			),
			nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
			nn.GELU(),
			# nn.Dropout2d(p=0.1),
			nn.Conv2d(
				out_channels,
				out_channels,
				kernel_size=config.kernel_size,
				padding=config.padding,
			),
			nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
			nn.GELU(),
			# nn.Dropout2d(p=0.1),
		)

	def forward(self, x, skip_x):
		x = self.upconv(x)
		# concatenate with skip connection
		x = torch.cat((x, skip_x), dim=1)
		x = self.conv(x)
		return x


class PatchEmbeddingConv(nn.Module):
	"""
	Patch embedding layer that uses a convolutional layer to extract patches
	and project each patch into an embedding vector.

	Args:
		config: ModelConfig
			Configuration object.
		patch_channels: int
			Number of channels in the patch. May change whether this is directly
			taking in data (ViT) or is deeper inside a model (TransUNet).

	Methods:
		forward(x: torch.Tensor) -> torch.Tensor
			Applies a convolutional layer to extract patches and project them into embedding space.
			Optionally adds positional embeddings.
	"""

	def __init__(self, config: ModelConfig, patch_channels: int):
		super().__init__()
		self.config: ModelConfig = config
		self.patch_channels: int = patch_channels
		# conv layer to extract patches and project them into embedding space
		self.conv: nn.Conv2d = nn.Conv2d(
			in_channels=self.patch_channels,
			out_channels=config.embed_dim,
			kernel_size=config.patch_size_tup,
			stride=config.patch_size_tup,
			bias=config.patch_conv_bias,
		)

		# if pos embeddings are made in patching
		# then create them here for both x and y dimensions
		if config.pos_embed_cls.at_patch_embed:
			self.pos_embed_x: PosEmbed = config.pos_embed_cls(
				config.embed_dim, config.pos_embed_max_n_axis
			)
			self.pos_embed_y: PosEmbed = config.pos_embed_cls(
				config.embed_dim, config.pos_embed_max_n_axis
			)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""returns (batch patches embed_dim)"""
		B, C, H, W = x.size()
		assert C == self.patch_channels, (
			f"Input channels mismatch: {C = } vs {self.patch_channels = }"
		)
		assert (
			H % self.config.patch_size_h == 0 and W % self.config.patch_size_w == 0
		), "Image dimensions must be divisible by the specified patch size."

		# apply conv layer
		# shape: (B, embed_dim, num_patches_h, num_patches_w)
		# Float[torch.Tensor, "batch embed_dim num_patches_h num_patches_w"]
		x = self.conv(x)

		B, embed_dim, num_patches_h, num_patches_w = x.size()
		assert embed_dim == self.config.embed_dim, (
			f"Embedding dimension mismatch: {embed_dim = } vs {self.config.embed_dim = }"
		)

		if self.config.pos_embed_cls.at_patch_embed:
			# create positional emebds
			# embed_dim num_patches_h
			pos_x = self.pos_embed_x(num_patches_w).to(x.device).unsqueeze(-2)
			# embed_dim, num_patches_w
			pos_y = self.pos_embed_y(num_patches_h).to(x.device).unsqueeze(-1)

			# print(f"{x.shape = }, {pos_x.shape = }, {pos_y.shape = }")
			# unsqueeze and add positional embeddings
			x = x + pos_x.unsqueeze(0) + pos_y.unsqueeze(0)

		# flatten along patch dim
		x = x.flatten(-2).transpose(
			-1, -2
		)  # shape: (B, num_patches = seq_length, embed_dim)

		return x, (num_patches_h, num_patches_w)


class DyT(nn.Module):
	"""
	Drop in replacement for Layernorm
	from new paper "Transformers without Normalization" by
	Zhu et al. Observes that layernorm outputs often look
	like tanh function, and this may be a faster alternative
	with similar performance.

	Args:
		C: int
			Number of channels.
		init_alpha: float
			Initial value for alpha parameter. Default set to
			0.5 as recommended in the paper.
	"""

	def __init__(self, C: int, init_alpha: float = 0.5):
		super(DyT, self).__init__()
		self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
		self.gamma = nn.Parameter(torch.ones(C))
		self.beta = nn.Parameter(torch.zeros(C))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = torch.tanh(self.alpha * x)
		return self.gamma * x + self.beta


class TransformerEncoder(nn.Module):
	"""
	Transformer encoder block.

	Args:
		config: ModelConfig
			Configuration object.

	Methods:
		forward(x: torch.Tensor) -> torch.Tensor
			Applies multi-head self-attention and feedforward network to the input tensor.
			Returns the transformed tensor.
	"""

	def __init__(self, config: ModelConfig):
		super(TransformerEncoder, self).__init__()
		self.config: ModelConfig = config
		self.dropout: nn.Dropout = nn.Dropout(config.dropout)

		# initialize the positional embedding
		self.pos_embed: PosEmbed | None
		# first, figure out if we do the pos embed at patching
		is_positional_embedding_at_patching: bool = (
			self.config.pos_embed_cls.at_patch_embed
		)
		# if we arent doing pos embed at patching, that must mean we are doing it here
		# (and probably using RoPE)
		if not is_positional_embedding_at_patching:
			# create the positional embedding
			self.pos_embed = config.pos_embed_cls(
				config.head_dim,  # creates vectors of the same size as the embeddings
				config.pos_embed_max_n_axis
				** 2,  # max number of patches in the axis is ^2 for RoPE
			)
		else:
			# it will be `None` if we are doing the pos embed at patching and not here
			self.pos_embed = None

		# old attention (just in case, does not work with RoPE)
		# self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

		# torchtune MHA used for RoPE
		self.attention: TT_MultiHeadAttention = TT_MultiHeadAttention(
			embed_dim=config.embed_dim,
			num_heads=config.num_heads,
			num_kv_heads=config.num_heads,  # for standard MHA, set kv heads to be same as heads
			attn_dropout=config.dropout,
			pos_embeddings=self.pos_embed,  # either RoPE or "None": happen at patching
			# initialize 'head_dim', 'q_proj', 'k_proj', 'v_proj', and 'output_proj'
			head_dim=config.head_dim,
			q_proj=nn.Linear(config.embed_dim, config.embed_dim, bias=config.head_bias),
			k_proj=nn.Linear(config.embed_dim, config.embed_dim, bias=config.head_bias),
			v_proj=nn.Linear(config.embed_dim, config.embed_dim, bias=config.head_bias),
			output_proj=nn.Linear(
				config.embed_dim, config.embed_dim, bias=config.head_bias
			),
			# batch_first=True,
		)
		self.norm1: nn.LayerNorm = nn.LayerNorm(config.embed_dim)

		self.mlp: nn.Sequential = nn.Sequential(
			nn.Linear(config.embed_dim, config.mlp_dim),
			nn.GELU(),
			nn.Linear(config.mlp_dim, config.embed_dim),
		)
		self.norm2: nn.LayerNorm = nn.LayerNorm(config.embed_dim)

	def forward(self, x):
		# attention
		attn_out = self.attention(x, x)
		x = x + self.dropout(attn_out)
		x = self.norm1(x)

		# mlp
		mlp_out = self.mlp(x)
		x = x + self.dropout(mlp_out)
		x = self.norm2(x)
		return x
