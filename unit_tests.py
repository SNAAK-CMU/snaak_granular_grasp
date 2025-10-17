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

        # Assert the type of the weight label
        assert isinstance(
            weight_label, torch.Tensor
        ), f"Weight label is not a tensor: {type(weight_label)}"
        assert (
            weight_label.dtype == torch.float32
        ), f"Weight label is not a float32 tensor: {weight_label.dtype}"
        assert (
            weight_label.shape == ()
        ), f"Weight label shape is incorrect: {weight_label.shape}"

    print("Dataset tested successfully!")


def unit_test_dataloaders(batch_size=2):
    train_loader, val_loader = create_train_val_dataloaders(
        data_dir="/home/parth/snaak/snaak_data/data_parth",
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    # Print the number of batches in the train and val loaders
    print(f"Train loader number of batches: {len(train_loader)}")
    print(f"Val loader number of batches: {len(val_loader)}")

    for i, ((rgb_patches, depth_patches), weight_labels) in enumerate(train_loader):
        assert rgb_patches.shape[1:] == (
            3,
            150,
            150,
        ), f"[Batch {i}] RGB patches shape is incorrect: {rgb_patches.shape[1:]}"
        assert depth_patches.shape[1:] == (
            1,
            150,
            150,
        ), f"[Batch {i}] Depth patches shape is incorrect: {depth_patches.shape[1:]}"
        assert (
            weight_labels.shape[1:] == ()
        ), f"[Batch {i}] Weight label shape is incorrect: {weight_labels}: {weight_labels.shape}"

    for i, ((rgb_patches, depth_patches), weight_labels) in enumerate(val_loader):
        assert rgb_patches.shape[1:] == (
            3,
            150,
            150,
        ), f"[Batch {i}] RGB patches shape is incorrect: {rgb_patches.shape[1:]}"
        assert depth_patches.shape[1:] == (
            1,
            150,
            150,
        ), f"[Batch {i}] Depth patches shape is incorrect: {depth_patches.shape[1:]}"
        assert (
            weight_labels.shape[1:] == ()
        ), f"[Batch {i}] Weight label shape is incorrect: {weight_labels}: {weight_labels.shape}"

    print("Dataloaders tested successfully!")


def unit_test_network():
    # Create data loaders
    train_loader, val_loader = create_train_val_dataloaders(
        data_dir="/home/parth/snaak/snaak_data/data_parth",
        batch_size=2,
        shuffle=True,
        num_workers=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MassEstimationModel().to(device)

    for i, ((rgb_patches, depth_patches), weight_labels) in enumerate(train_loader):
        rgb_patches = rgb_patches.to(device)
        depth_patches = depth_patches.to(device)
        weight_labels = weight_labels.to(device)

        predicted_mass = model(rgb_patches, depth_patches)
        assert (
            predicted_mass.shape[:-1] == weight_labels.shape
        ), f"[Batch {i}] Predicted mass shape is incorrect: {predicted_mass.shape} != {weight_labels.shape}"


if __name__ == "__main__":
    unit_test_dataset()
    unit_test_dataloaders()
    unit_test_network()
