import torch
import torch.nn as nn

class Dense_Block(nn.Module):
    def __init__(self, channels, num_layers=4, growth_rate=16, kernel_size=3):
        super(Dense_Block, self).__init__()
        self.growth_rate = growth_rate
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(channels, growth_rate, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.ReLU(inplace=True)
                )
            )
            channels += growth_rate
        self.compress = nn.Conv2d(channels, channels - num_layers * growth_rate, kernel_size=1, padding=0)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return self.compress(torch.cat(features, dim=1))
