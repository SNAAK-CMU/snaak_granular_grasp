import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with Conv2d, BatchNorm, and ReLU activation."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.block(x)


class ImageBranch(nn.Module):
    """Branch for processing either RGB or depth images."""

    def __init__(self, in_channels=3, num_blocks=5, base_channels=16):
        super(ImageBranch, self).__init__()

        channels = [in_channels] + [base_channels] * num_blocks
        self.layers = []
        for i in range(num_blocks):
            self.layers.append(ConvBlock(channels[i], channels[i + 1]))
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class MassEstimationModel(nn.Module):
    """
    Neural network for estimating grasp mass from RGB and depth images.

    Architecture:
    - Dual-stream CNN processing RGB and depth images separately
    - 5 repeated ConvBlocks for each branch
    - Feature concatenation
    - Fully connected layers for regression
    """

    def __init__(
        self,
        rgb_channels=3,
        depth_channels=1,
        base_channels=16,
        fc_hidden=128,
        fc_hidden2=64,
        num_blocks=4,
    ):
        super(MassEstimationModel, self).__init__()

        # RGB image branch
        self.rgb_branch = ImageBranch(rgb_channels, num_blocks, base_channels)

        # Depth image branch
        self.depth_branch = ImageBranch(depth_channels, num_blocks, base_channels)

        # Calculate the size after conv blocks and maxpool
        # Assuming input size of 50x50, after 4 conv blocks and 1 maxpool:
        # Over 5 blocks, output size: 50 (input size) -> 25 -> 12 -> 6 -> 3 (output size)
        conv_output_size = 3 * 3 * base_channels * 2  # *2 for concatenation

        # Fully connected layers
        self.combined_branch = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),
        )

    def forward(self, rgb_img, depth_img):
        """
        Forward pass through the network.

        Args:
            rgb_img: RGB image tensor of shape (batch_size, 3, 150, 150)
            depth_img: Depth image tensor of shape (batch_size, 1, 150, 150)

        Returns:
            mass_estimate: Estimated grasp mass tensor of shape (batch_size, 1)
        """
        # Process RGB and depth images through separate branches
        rgb_features = self.rgb_branch(rgb_img)
        depth_features = self.depth_branch(depth_img)

        # Flatten features
        rgb_features = torch.flatten(rgb_features, start_dim=1)
        depth_features = torch.flatten(depth_features, start_dim=1)

        # Concatenate features from both branches along the feature dimension
        combined_features = torch.cat([rgb_features, depth_features], dim=1)

        # Pass through fully connected layers
        mass_estimate = self.combined_branch(combined_features)
        return mass_estimate


if __name__ == "__main__":
    # Test the model
    model = MassEstimationModel()

    # Create dummy inputs
    batch_size = 4
    rgb_input = torch.randn(batch_size, 3, 150, 150)
    depth_input = torch.randn(batch_size, 1, 150, 150)

    # Forward pass
    output = model(rgb_input, depth_input)

    print(f"Model created successfully!")
    print(f"Input shapes: RGB {rgb_input.shape}, Depth {depth_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test loss function
    true_mass = torch.randn(batch_size, 1)
    loss = F.mse_loss(output, true_mass)
    print(f"MSE Loss: {loss.item():.4f}")
