import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

from network import MassEstimationModel
from train import create_train_val_dataloaders
from data_utils import create_transform_rgb, create_transform_depth
from data_utils import CROP_XMIN, CROP_XMAX, CROP_YMIN, CROP_YMAX
from data_utils import (
    BIN_LENGTH_PIX,
    BIN_WIDTH_PIX,
    BIN_WIDTH_M,
    BIN_LENGTH_M,
    BIN_HEIGHT,
    WINDOW_SIZE,
)

BIN_PADDING = 50
DEBUG_PLOT = True


def _tensor_to_numpy_image(t: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor image to a numpy array suitable for plotting.

    Handles both RGB (C,H,W) and depth (1,H,W) tensors.
    """
    t_cpu = t.detach().cpu()
    if t_cpu.ndim == 3 and t_cpu.shape[0] in (1, 3):
        # (C,H,W) -> (H,W,C)
        arr = t_cpu.numpy()
        arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        return arr
    if t_cpu.ndim == 2:
        return t_cpu.numpy()
    # Fallback: return as numpy
    return t_cpu.numpy()


def _plot_and_save_sample(
    rgb_tensor: torch.Tensor,
    depth_tensor: torch.Tensor,
    y_true: float,
    y_pred: float,
    out_path: str,
) -> None:
    """
    Create a side-by-side plot of RGB and depth with predicted and true labels and save it.
    """
    rgb_img = _tensor_to_numpy_image(rgb_tensor)
    depth_img = _tensor_to_numpy_image(depth_tensor)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # RGB
    axes[0].imshow(np.clip(rgb_img, 0.0, 1.0))
    axes[0].axis("off")
    axes[0].set_title("RGB")

    # Depth
    im = axes[1].imshow(depth_img, cmap="viridis")
    axes[1].axis("off")
    axes[1].set_title("Depth")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f"Pred: {y_pred:.3f} | True: {y_true:.3f}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_inference_and_save_plots(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    output_dir: str,
    device: torch.device,
    limit_batches=None,
) -> None:
    """
    Run inference on a dataloader and save per-sample plots to a directory.

    Args:
        model: Trained model that takes (rgb_batch, depth_batch) -> predictions of shape (B,1) or (B,).
        data_loader: Dataloader yielding ((rgb_batch, depth_batch), labels).
        output_dir: Directory where plots will be saved.
        device: Torch device to run on; if None, inferred from model or CUDA availability.
        limit_batches: If set, process at most this many batches (useful for quick tests).
    """
    # Resolve device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    os.makedirs(output_dir, exist_ok=True)

    global_sample_idx = 0
    processed_batches = 0

    with torch.no_grad():
        for batch_idx, ((rgb_batch, depth_batch), labels) in enumerate(data_loader):
            rgb_batch = rgb_batch.to(device)
            depth_batch = depth_batch.to(device)
            labels = labels.to(device)

            preds = model(rgb_batch, depth_batch)
            preds = preds.squeeze(-1)  # (B,1) -> (B,) if needed

            batch_size = preds.shape[0]

            for i in range(batch_size):
                y_pred = float(preds[i].detach().cpu().item())
                y_true = float(labels[i].detach().cpu().item())

                out_path = os.path.join(
                    output_dir, f"sample_{global_sample_idx:06d}.png"
                )
                _plot_and_save_sample(
                    rgb_tensor=rgb_batch[i].detach().cpu(),
                    depth_tensor=depth_batch[i].detach().cpu(),
                    y_true=y_true,
                    y_pred=y_pred,
                    out_path=out_path,
                )

                global_sample_idx += 1

            processed_batches += 1
            if limit_batches is not None and processed_batches >= limit_batches:
                break

    print(
        f"Inference complete. Saved {global_sample_idx} plots to: {os.path.abspath(output_dir)}"
    )


def get_patches_from_image_and_depth_maps(
    rgb_img, depth_img, n_patch_length=7, n_patch_width=5, bin_padding=50
):
    """
    Sample 7 points along the bin's length (x axis) and 5 points along the width (y axis),
    equally spaced. Extract centered patches around each point.

    Images given to this function are in OpenCV format:
      - rgb_img shape: (BIN_LENGTH_PIX, BIN_WIDTH_PIX, 3), dtype=uint8, BGR order.
      - depth_img shape: (BIN_LENGTH_PIX, BIN_WIDTH_PIX), dtype=float32 or uint16.
    """

    assert rgb_img.shape == (
        BIN_LENGTH_PIX,
        BIN_WIDTH_PIX,
        3,
    ), f"RGB image shape is incorrect: {rgb_img.shape}"
    assert depth_img.shape == (
        BIN_LENGTH_PIX,
        BIN_WIDTH_PIX,
    ), f"Depth image shape is incorrect: {depth_img.shape}"
    assert (
        rgb_img.shape[0:2] == depth_img.shape[0:2]
    ), "RGB and depth image shapes do not match"

    # Compute coordinates for patch centers (y, x)
    # Length = rows (Y), Width = columns (X) in image patch
    padding = bin_padding
    x_vals = np.linspace(padding, BIN_WIDTH_PIX - padding, n_patch_width)
    y_vals = np.linspace(padding, BIN_LENGTH_PIX - padding, n_patch_length)

    # Convert to int and clamp within safe valid range
    x_vals = np.round(x_vals).astype(int)
    y_vals = np.round(y_vals).astype(int)

    patches_rgb = []
    patches_depth = []
    centers = []

    for y in y_vals:
        for x in x_vals:
            top = y - WINDOW_SIZE // 2
            left = x - WINDOW_SIZE // 2
            rgb_patch = rgb_img[top : top + WINDOW_SIZE, left : left + WINDOW_SIZE, :]
            depth_patch = depth_img[top : top + WINDOW_SIZE, left : left + WINDOW_SIZE]
            patches_rgb.append(rgb_patch)
            patches_depth.append(depth_patch)
            centers.append((x, y))

    # Make a copy so the original image isn't modified (OpenCV format, uint8)
    rgb_img_for_viz = rgb_img.copy()

    if DEBUG_PLOT:
        for center in centers:
            x, y = center
            # Top-left and bottom-right of the patch
            top_left = (x - WINDOW_SIZE // 2, y - WINDOW_SIZE // 2)
            bottom_right = (x + WINDOW_SIZE // 2 - 1, y + WINDOW_SIZE // 2 - 1)
            # Draw rectangle (BGR=green)
            cv2.rectangle(rgb_img_for_viz, top_left, bottom_right, (0, 255, 0), 2)
            # Draw center (BGR=red)
            cv2.circle(rgb_img_for_viz, (x, y), 4, (0, 0, 255), -1)

        # Optional: Show or save the visualization (commented out, for user to enable as needed)
        cv2.imshow("Patch Visualization", rgb_img_for_viz)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return patches_rgb, patches_depth, centers


def infer_on_bin(rgb_img, depth_img, model, device):

    # Crop out the bin from the image and depth maps
    rgb_img = rgb_img[CROP_YMIN:CROP_YMAX, CROP_XMIN:CROP_XMAX, :]
    depth_img = depth_img[CROP_YMIN:CROP_YMAX, CROP_XMIN:CROP_XMAX]

    # Get patches from the image and depth maps
    patches_rgb, patches_depth, centers = get_patches_from_image_and_depth_maps(
        rgb_img, depth_img, bin_padding=BIN_PADDING
    )

    # Preprocess the image and depth maps
    transform_rgb = create_transform_rgb()
    transform_depth = create_transform_depth()
    pred_weights = []
    for rgb_patch, depth_patch, center in zip(patches_rgb, patches_depth, centers):
        rgb_patch = transform_rgb(rgb_patch)
        depth_patch = transform_depth(depth_patch)
        rgb_patch = rgb_patch.to(torch.float32).to(device)
        depth_patch = depth_patch.to(torch.float32).to(device)

        rgb_patch = rgb_patch.unsqueeze(0)
        depth_patch = depth_patch.unsqueeze(0)

        pred = model(rgb_patch, depth_patch)
        pred = pred.squeeze(-1)

        pred_weights.append(pred.item())

    if DEBUG_PLOT:
        resize_factor = 2
        rgb_img_for_viz = cv2.resize(
            rgb_img.copy(),
            fx=resize_factor,
            fy=resize_factor,
            interpolation=cv2.INTER_NEAREST,
            dsize=None,
        )
        depth_img_for_viz = cv2.resize(
            depth_img.copy(),
            fx=resize_factor,
            fy=resize_factor,
            interpolation=cv2.INTER_NEAREST,
            dsize=None,
        )
        for center, pred_weight in zip(centers, pred_weights):
            x, y = center
            x = x * resize_factor
            y = y * resize_factor
            # Top-left and bottom-right of the patch
            # top_left = (x - WINDOW_SIZE // 2, y - WINDOW_SIZE // 2)
            # bottom_right = (x + WINDOW_SIZE // 2 - 1, y + WINDOW_SIZE // 2 - 1)
            # Draw rectangle (BGR=green)
            # cv2.rectangle(rgb_img_for_viz, top_left, bottom_right, (0, 255, 0), 2)
            # Draw center (BGR=red)
            # cv2.circle(rgb_img_for_viz, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(
                rgb_img_for_viz,
                f"{pred_weight:.0f}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

        cv2.imshow(
            "Predicted Weights",
            rgb_img_for_viz,
        )

        # Show the depth image using matplotlib
        plt.imshow(depth_img_for_viz, cmap="viridis")
        plt.colorbar()
        plt.show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        plt.close()


def main():
    model_path = "/home/parth/snaak/projects/granular_grasp/runs/train_w50_run_1/mass_estimation_model.pth"
    input_data_dir = "/home/parth/snaak/snaak_data/data_parth"
    output_data_dir = (
        "/home/parth/snaak/projects/granular_grasp/runs/train_w50_run_1/inference"
    )
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MassEstimationModel()
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model.eval()
    model.to(device)

    train_loader, val_loader = create_train_val_dataloaders(
        data_dir=input_data_dir,
        batch_size=batch_size,
    )

    run_inference_and_save_plots(
        model=model,
        data_loader=val_loader,
        output_dir=output_data_dir,
        device=device,
    )


def test_get_patches_from_image_and_depth_maps():
    img_path = "/home/parth/snaak/projects/granular_grasp/rgb_image.jpg"
    img = cv2.imread(img_path)

    # Create a depth map
    depth_map = np.load("/home/parth/snaak/projects/granular_grasp/depth_map.npy")

    # Crop out the bin from the image and depth maps
    img = img[CROP_YMIN:CROP_YMAX, CROP_XMIN:CROP_XMAX, :]
    depth_map = depth_map[CROP_YMIN:CROP_YMAX, CROP_XMIN:CROP_XMAX]

    get_patches_from_image_and_depth_maps(img, depth_map, bin_padding=BIN_PADDING)


def test_infer_on_bin():

    model_path = "/home/parth/snaak/projects/granular_grasp/runs/train_w50_run_1/mass_estimation_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MassEstimationModel()
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model.eval()
    model.to(device)

    img_path = "/home/parth/snaak/projects/granular_grasp/rgb_image.jpg"
    img = cv2.imread(img_path)
    depth_map = np.load("/home/parth/snaak/projects/granular_grasp/depth_map.npy")
    infer_on_bin(img, depth_map, model, device)


if __name__ == "__main__":
    test_infer_on_bin()
