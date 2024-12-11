import torch.nn as nn
import torch.nn.functional as F
class Cnn_Block(nn.Module):
	def __init__(self, in_channels, out_channels, strides, use_1x1_conv=True):
		super(Cnn_Block, self).__init__()
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
		X = self.blk(X)
		return F.relu(X)