import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import os
import datetime
import json

from data_utils import GraspDataset, create_transform_rgb, create_transform_depth
from network import MassEstimationModel


def create_run_directory(base_dir, run_name=None):
    """
    Create a new directory for this training run.

    Args:
        base_dir: Base directory to create runs in

    Returns:
        str: Path to the created run directory
    """
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # Generate timestamp for unique run name
    if run_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_dir, f"run_{timestamp}")
    else:
        run_dir = os.path.join(base_dir, run_name)

    # Create the run directory
    os.makedirs(run_dir, exist_ok=True)

    print(f"Created run directory: {run_dir}")
    return run_dir


def save_training_config(run_dir, config):
    """
    Save training configuration to JSON file.

    Args:
        run_dir: Directory to save config in
        config: Dictionary containing training configuration
    """
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Training configuration saved to {config_path}")


def create_train_val_dataloaders(
    ingredient_name, data_dir, batch_size=8, shuffle=True, num_workers=2
):
    """
    Create a DataLoader for the grasp dataset.

    Args:
        data_dir: Path to the data directory
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading

    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataset = GraspDataset(ingredient_name, data_dir)

    # Split into train and validation sets (80/20 split)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,  # Drop the last incomplete batch
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,  # Drop the last incomplete batch
    )

    return train_dataloader, val_dataloader


def train_single_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train the model for a single epoch.

    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        optimizer: Optimizer for updating model parameters
        criterion: Loss function
        device: Device to run training on (CPU/GPU)

    Returns:
        float: Average loss for the epoch
    """
    model.train()  # Set model to training mode
    total_loss = 0.0
    num_batches = 0

    for batch_idx, ((rgb_patches, depth_patches), weight_labels) in enumerate(
        train_loader
    ):
        # Move data to device
        rgb_patches = rgb_patches.to(device)
        depth_patches = depth_patches.to(device)
        weight_labels = weight_labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        predicted_mass = model(rgb_patches, depth_patches)

        # Calculate loss
        loss = criterion(predicted_mass, weight_labels.unsqueeze(1))

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1

        # Print progress
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    # Calculate average loss
    avg_loss = total_loss / num_batches
    return avg_loss


def validate_single_epoch(model, val_loader, criterion, device):
    """
    Validate the model for a single epoch.

    Args:
        model: The neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run validation on (CPU/GPU)

    Returns:
        float: Average validation loss for the epoch
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # Disable gradient computation for validation
        for batch_idx, ((rgb_patches, depth_patches), weight_labels) in enumerate(
            val_loader
        ):
            # Move data to device
            rgb_patches = rgb_patches.to(device)
            depth_patches = depth_patches.to(device)
            weight_labels = weight_labels.to(device)

            # Forward pass
            predicted_mass = model(rgb_patches, depth_patches)

            # Calculate loss
            loss = criterion(predicted_mass, weight_labels.unsqueeze(1))

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

    # Calculate average loss
    avg_loss = total_loss / num_batches
    return avg_loss


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=50,
    learning_rate=0.001,
    device="cpu",
    run_dir=None,
):
    """
    Train the model for multiple epochs.

    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        device: Device to run training on
        run_dir: Directory to save training outputs

    Returns:
        tuple: (train_losses, val_losses) - Lists of average losses per epoch
    """
    # Move model to device
    model = model.to(device)

    # Define optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = F.mse_loss

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Track losses
    train_losses = []
    val_losses = []

    print(f"Starting training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {num_epochs} epochs")
    print("-" * 50)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Train for one epoch
        train_avg_loss = train_single_epoch(
            model, train_loader, optimizer, criterion, device
        )
        print(f"Train Average Loss: {train_avg_loss:.4f}")

        # Validate for one epoch
        val_avg_loss = validate_single_epoch(model, val_loader, criterion, device)
        print(f"Validation Average Loss: {val_avg_loss:.4f}")

        # Update learning rate
        scheduler.step(val_avg_loss)

        # Store losses
        train_losses.append(train_avg_loss)
        val_losses.append(val_avg_loss)

        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.10f}")
        print("-" * 50)

    return train_losses, val_losses


def plot_training_curves(train_losses, val_losses=None, save_path=None):
    """
    Plot training curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch (optional)
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", color="blue")

    if val_losses is not None:
        plt.plot(val_losses, label="Validation Loss", color="red")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training curve saved to {save_path}")

    # plt.show()


def save_model(model, filepath):
    """
    Save the trained model.

    Args:
        model: The trained model
        filepath: Path to save the model
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_architecture": model.__class__.__name__,
        },
        filepath,
    )
    print(f"Model saved to {filepath}")


def main():
    """
    Main training function.
    """
    # Configuration
    data_dir = "/home/parth/snaak/snaak_data/data_parth"
    base_dir = "/home/parth/snaak/projects/granular_grasp/runs"
    run_name = "train_w50_run_8"
    batch_size = 8
    num_epochs = 100
    learning_rate = 0.0005

    # Create run directory
    run_dir = create_run_directory(base_dir, run_name)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loader
    print("Creating data loader...")
    train_loader, val_loader = create_train_val_dataloaders(
        data_dir, batch_size=batch_size
    )
    print(f"Train dataset size: {len(train_loader.dataset)} samples")
    print(f"Number of batches: {len(train_loader)}")
    print(f"Validation dataset size: {len(val_loader.dataset)} samples")
    print(f"Number of batches: {len(val_loader)}")
    print(f"Batch size: {batch_size}")

    # Save training configuration
    config = {
        "data_dir": data_dir,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "run_name": run_name,
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
    }
    save_training_config(run_dir, config)

    # Create model
    print("Creating model...")
    model = MassEstimationModel()

    # Train model
    print("Starting training...")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        run_dir=run_dir,
    )

    # Plot training curves
    plot_path = os.path.join(run_dir, "training_curves.png")
    plot_training_curves(train_losses, val_losses, save_path=plot_path)

    # Save model
    model_path = os.path.join(run_dir, "mass_estimation_model.pth")
    save_model(model, model_path)

    # Save losses to JSON
    losses_data = {"train_losses": train_losses, "val_losses": val_losses}
    losses_path = os.path.join(run_dir, "losses.json")
    with open(losses_path, "w") as f:
        json.dump(losses_data, f, indent=4)
    print(f"Losses saved to {losses_path}")

    print(f"Training completed! All outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
