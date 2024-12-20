import torch.nn as nn
import torch.nn.functional as F
class Residual_Block(nn.Module):
	def __init__(self, in_channels, out_channels, strides, use_1x1_conv=True):
		super(Residual_Block, self).__init__()

		self.use_1x1_conv = use_1x1_conv
		self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
		self.blk = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.PReLU(),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
		)

	def forward(self, X):
		"""
		:param X: tensor with shape (N, C, H, W)
		"""
		X_original = X.clone()
		X = self.blk(X)
		if self.use_1x1_conv:
			X_original = self.conv1x1(X_original)

		return F.relu(X + X_original)