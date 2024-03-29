import torch
import torch.nn as nn

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, padding:int=2, stride:int=1) -> None:
        super(ConvBlock, self).__init__()

        padding = (kernel_size - 1) // 2 # Overwrite the pading for now, i dont see how they keep the image size and use padding 2 in the paper

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.relu = nn.ReLU()
        self.bn   = nn.BatchNorm2d(num_features=out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class TrackNet(torch.nn.Module):
    """ 
    Using the architecture described in:
    TrackNet: A deep learning network for tracking high-speed and tiny objects in sports application
    https://findit.dtu.dk/en/catalog/5e061db2d9001d57e3218aae
    """
    def __init__(self, in_features: int, out_features: int = 1, take_sigmoid: bool = True, weights_path: str | None = None) -> None:
        """ """
        super(TrackNet, self).__init__()

        self.name = f"TrackNet_{in_features}"
        self.take_sigmoid = take_sigmoid
        
        self.sigmoid = nn.Sigmoid()
        self.downsample_block = nn.Sequential(
            ConvBlock(in_channels=in_features, out_channels=64),
            ConvBlock(in_channels=64, out_channels=64),
            
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            ConvBlock(in_channels=64, out_channels=128),
            ConvBlock(in_channels=128, out_channels=128),

            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            ConvBlock(in_channels=128, out_channels=256),
            ConvBlock(in_channels=256, out_channels=256),
            ConvBlock(in_channels=256, out_channels=256),

            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            ConvBlock(in_channels=256, out_channels=512),
            ConvBlock(in_channels=512, out_channels=512),
            ConvBlock(in_channels=512, out_channels=512),
        )

        self.upsample_block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock(in_channels=512, out_channels=256),
            ConvBlock(in_channels=256, out_channels=256),
            ConvBlock(in_channels=256, out_channels=256),

            nn.Upsample(scale_factor=2),
            ConvBlock(in_channels=256, out_channels=128),
            ConvBlock(in_channels=128, out_channels=128),

            nn.Upsample(scale_factor=2),
            ConvBlock(in_channels=128, out_channels=64),
            ConvBlock(in_channels=64, out_channels=64),
            ConvBlock(in_channels=64, out_channels=out_features)
        )

        if weights_path:
            self.load_state_dict(torch.load(weights_path, map_location="cpu"))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ """
        x = self.downsample_block(x)
        x = self.upsample_block(x)
        return self.sigmoid(x) if self.take_sigmoid else x


if __name__ == "__main__":
    a = TrackNet(3)
    x = torch.randn(4,3,640,360)
    print(a(x).shape)