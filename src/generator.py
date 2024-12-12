import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CONFIG import FEATURE_EXTRACTOR
import torch
from Residual_Block import Residual_Block
from Cnn_Block import Cnn_Block
from Dense_Block import Dense_Block
from SwinTransformer_Block import SwinTransformer_Block


class Generator(nn.Module):
	def __init__(self, n_blks, upscale_factor=4):
		super(Generator, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
		self.prelu1 = nn.PReLU()

		self.blocks = nn.Sequential()
		if FEATURE_EXTRACTOR == "Cnn_Block":
			for i in range(n_blks):
				self.blocks.add_module(
					f"blk_{i}",
					Cnn_Block(in_channels=64, out_channels=64, strides=1, use_1x1_conv=False),
				)
		elif FEATURE_EXTRACTOR == "Dense_Block":
			for i in range(n_blks):
				self.blocks.add_module(
					f"blk_{i}",
					Dense_Block(channels=64, num_layers=4, growth_rate=16),
				)
		elif FEATURE_EXTRACTOR == "SwinTransformer_Block":
			for i in range(n_blks):
				self.blocks.add_module(
					f"blk_{i}",
					SwinTransformer_Block(dim=64, num_heads=8, window_size=7),
				)
		elif FEATURE_EXTRACTOR == "Residual_Block":
			for i in range(n_blks):
				self.blocks.add_module(
					f"blk_{i}",
					Residual_Block(in_channels=64, out_channels=64, strides=1, use_1x1_conv=False),
				)
		else:
			raise ValueError(f"Unknown FEATURE_EXTRACTOR: {FEATURE_EXTRACTOR}")

		self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
		self.bn = nn.BatchNorm2d(64)

		self.pixel_shufflers = nn.Sequential()
		for i in range(int(np.log2(upscale_factor))):
			self.pixel_shufflers.add_module(f"pixel_shuffle_blk_{i}",
											PixelShufflerBlock(in_channels=64, upscale_factor=2))
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, padding=4)

	def forward(self, X):
		X = self.prelu1(self.conv1(X))
		X_before_blks = X.clone()
		X = self.blocks(X)
		X = self.bn(self.conv2(X))
		X = F.relu(X + X_before_blks)

		X = self.pixel_shufflers(X)
		X = self.conv3(X)

		return F.tanh(X)


class PixelShufflerBlock(nn.Module):
	def __init__(self, in_channels, upscale_factor=2):
		super(PixelShufflerBlock, self).__init__()

		self.blk = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=3, padding=1),
			nn.PixelShuffle(upscale_factor=upscale_factor),
			nn.PReLU()
		)

	def forward(self, X):
		return self.blk(X)