import torch

from data_utils import GraspDataset
from train import create_train_val_dataloaders
from network import MassEstimationModel


def unit_test_dataset():
    dataset = GraspDataset(
        data_dir="/home/parth/snaak/snaak_data/data_parth",
    )
    for i in range(len(dataset)):
        (rgb_patches, depth_patches), weight_label = dataset[i]
        # Assert the shapes of the rgb and depth patches
        assert rgb_patches.shape == (
            3,
            150,
            150,
        ), f"RGB patches shape is incorrect: {rgb_patches.shape}"
        assert depth_patches.shape == (
            1,
            150,
            150,
        ), f"Depth patches shape is incorrect: {depth_patches.shape}"
        assert weight_label.shape == (), "Weight label shape is incorrect"
    print("Dataset tested successfully!")


def unit_test_dataloaders(batch_size=2):
    train_loader, val_loader = create_train_val_dataloaders(
        data_dir="/home/parth/snaak/snaak_data/data_parth",
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    for (rgb_patches, depth_patches), weight_label in train_loader:
        assert rgb_patches.shape[1:] == (
            3,
            150,
            150,
        ), f"RGB patches shape is incorrect: {rgb_patches.shape[1:]}"
        assert depth_patches.shape[1:] == (
            1,
            150,
            150,
        ), f"Depth patches shape is incorrect: {depth_patches.shape[1:]}"
        assert weight_label.shape == (), "Weight label shape is incorrect"

    for (rgb_patches, depth_patches), weight_label in val_loader:
        assert rgb_patches.shape[1:] == (
            3,
            150,
            150,
        ), f"RGB patches shape is incorrect: {rgb_patches.shape[1:]}"
        assert depth_patches.shape[1:] == (
            1,
            150,
            150,
        ), f"Depth patches shape is incorrect: {depth_patches.shape[1:]}"
        assert weight_label.shape == (), "Weight label shape is incorrect"

    print("Dataloaders tested successfully!")


def unit_test_network():
    # Create data loaders
    train_loader, val_loader = create_train_val_dataloaders(
        data_dir="/home/parth/snaak/snaak_data/data_parth",
        batch_size=2,
        shuffle=True,
        num_workers=2,
    )

    # Print sample size of the data loaders
    print(f"Train loader sample size: {len(train_loader.dataset)}")
    print(f"Val loader sample size: {len(val_loader.dataset)}")
    print(f"Train loader batch size: {len(train_loader)}")
    print(f"Val loader batch size: {len(val_loader)}")

    (rgb_patches, depth_patches), weight_labels = train_loader.dataset[0]
    print(f"RGB patches shape: {rgb_patches.shape}")
    print(f"Depth patches shape: {depth_patches.shape}")
    print(f"Weight labels shape: {weight_labels.shape}")
    print(f"Weight labels: {weight_labels}")


if __name__ == "__main__":
    unit_test_dataset()
