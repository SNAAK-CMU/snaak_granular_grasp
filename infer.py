import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm

from network import MassEstimationModel
from train import create_train_val_dataloaders
from data_utils import GraspDataset, create_transform_rgb, create_transform_depth
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
        centers_row = []
        patches_rgb_row = []
        patches_depth_row = []
        for x in x_vals:
            top = y - WINDOW_SIZE // 2
            left = x - WINDOW_SIZE // 2
            rgb_patch = rgb_img[top : top + WINDOW_SIZE, left : left + WINDOW_SIZE, :]
            depth_patch = depth_img[top : top + WINDOW_SIZE, left : left + WINDOW_SIZE]
            patches_rgb_row.append(rgb_patch)
            patches_depth_row.append(depth_patch)
            centers_row.append((x, y))
        centers.append(centers_row)
        patches_rgb.append(patches_rgb_row)
        patches_depth.append(patches_depth_row)

    # Make a copy so the original image isn't modified (OpenCV format, uint8)
    rgb_img_for_viz = rgb_img.copy()

    if DEBUG_PLOT and False:
        for centers_row in centers:
            for center in centers_row:
                x, y = center
                # Top-left and bottom-right of the patch
                top_left = (x - WINDOW_SIZE // 2, y - WINDOW_SIZE // 2)
                bottom_right = (x + WINDOW_SIZE // 2 - 1, y + WINDOW_SIZE // 2 - 1)
                # Draw rectangle (BGR=green)
                cv2.rectangle(rgb_img_for_viz, top_left, bottom_right, (0, 255, 0), 2)
                # Draw center (BGR=red)
                cv2.circle(rgb_img_for_viz, (x, y), 4, (0, 0, 255), -1)

        cv2.imshow("Patch Visualization", rgb_img_for_viz)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return patches_rgb, patches_depth, centers


def infer_on_bin(rgb_img, depth_img, model, transform_rgb, transform_depth, device):
    # Crop out the bin from the image and depth maps
    rgb_img = rgb_img[CROP_YMIN:CROP_YMAX, CROP_XMIN:CROP_XMAX, :]
    depth_img = depth_img[CROP_YMIN:CROP_YMAX, CROP_XMIN:CROP_XMAX]

    # Get patches from the image and depth maps
    patches_rgb, patches_depth, centers = get_patches_from_image_and_depth_maps(
        rgb_img, depth_img, bin_padding=BIN_PADDING
    )

    # print("Got patches from the image and depth maps")
    # print(f"Number of patches: {len(patches_rgb)}")
    # print(f"Number of patches per row: {len(patches_rgb)}")
    # print(f"Number of patches per column: {len(patches_rgb[0])}")
    # print(f"Shape of each patch: {patches_rgb[0][0].shape}")

    # Run inference on the patches
    pred_weights = []
    for row_i in range(len(patches_rgb)):
        pred_weights_row = []
        for col_i in range(len(patches_rgb[row_i])):
            rgb_patch = patches_rgb[row_i][col_i]
            depth_patch = patches_depth[row_i][col_i]

            # Preprocess the patch
            rgb_patch = transform_rgb(rgb_patch)
            depth_patch = transform_depth(depth_patch)
            rgb_patch = rgb_patch.to(torch.float32).to(device)
            depth_patch = depth_patch.to(torch.float32).to(device)

            rgb_patch = rgb_patch.unsqueeze(0)
            depth_patch = depth_patch.unsqueeze(0)

            pred = model(rgb_patch, depth_patch)
            pred = pred.squeeze(-1)

            pred_weights_row.append(pred.item())
        pred_weights.append(pred_weights_row)

    centers = np.array(centers)
    pred_weights = np.array(pred_weights)

    if DEBUG_PLOT and False:
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
        # Handle 2D arrays of centers and pred_weights
        for i in range(centers.shape[0]):
            for j in range(centers.shape[1]):
                x, y = centers[i, j]
                pred_weight = pred_weights[i, j]
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
                    (int(x), int(y)),
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

    return centers, pred_weights


def calculate_loss(row_i, col_i, w_desired, pred_weights, lambda_neighbor=0.25):
    cell_loss = abs(pred_weights[row_i][col_i] - w_desired)
    neighbor_loss = 0
    n_neighbors = 0
    for neighbor_row in [row_i - 1, row_i + 1]:
        for neighbor_col in [col_i - 1, col_i + 1]:
            # skip if neighbor is out of bounds
            if not (
                0 <= neighbor_row < pred_weights.shape[0]
                and 0 <= neighbor_col < pred_weights.shape[1]
            ):
                continue

            neighbor_loss += abs(pred_weights[neighbor_row][neighbor_col] - w_desired)
            n_neighbors += 1
    neighbor_loss = neighbor_loss / n_neighbors

    total_loss = cell_loss + lambda_neighbor * neighbor_loss
    return total_loss


def get_xy_for_weight(
    w_desired,
    rgb_img,
    depth_img,
    model,
    device,
    transform_rgb,
    transform_depth,
    save_path=None,
):

    centers, pred_weights = infer_on_bin(
        rgb_img, depth_img, model, transform_rgb, transform_depth, device
    )

    # Crop out the bin from the image and depth maps
    rgb_img = rgb_img[CROP_YMIN:CROP_YMAX, CROP_XMIN:CROP_XMAX, :]
    depth_img = depth_img[CROP_YMIN:CROP_YMAX, CROP_XMIN:CROP_XMAX]

    # Calculate a score for each point on the grid based on the desired weight
    best_x, best_y, min_loss = 0, 0, float("inf")
    losses = np.zeros((centers.shape[0], centers.shape[1])) * 1000.0
    for row_i in range(centers.shape[0]):
        for col_i in range(centers.shape[1]):
            loss = calculate_loss(row_i, col_i, w_desired, pred_weights)
            losses[row_i, col_i] = loss
            if loss < min_loss:
                min_loss = loss
                best_x, best_y = centers[row_i, col_i]

    # print(f"Best x: {best_x}, Best y: {best_y}, Min loss: {min_loss}")

    if DEBUG_PLOT:
        resize_factor = 2
        rgb_img_for_viz = cv2.resize(
            rgb_img.copy(),
            fx=resize_factor,
            fy=resize_factor,
            interpolation=cv2.INTER_NEAREST,
            dsize=None,
        )

        # Draw a circle at the best x, y
        best_x_print = best_x * resize_factor
        best_y_print = best_y * resize_factor
        cv2.circle(
            rgb_img_for_viz,
            (int(best_x_print), int(best_y_print)),
            15,
            (255, 0, 255),
            2,
        )

        # Handle 2D arrays of centers and pred_weights
        for i in range(centers.shape[0]):
            for j in range(centers.shape[1]):
                x, y = centers[i, j]
                pred_weight = pred_weights[i, j]
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
                    f"{pred_weight:.1f}",
                    (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    rgb_img_for_viz,
                    f"{losses[i, j]:.1f}",
                    (int(x), int(y) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

        if save_path is not None:
            cv2.imwrite(save_path, rgb_img_for_viz)
        else:
            cv2.imshow(
                "Predicted Weights",
                rgb_img_for_viz,
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            plt.close()

    return best_x, best_y


def main():
    model_path = "/home/parth/snaak/projects/granular_grasp/runs/train_w50_run_5/mass_estimation_model.pth"
    input_data_dir = "/home/parth/snaak/snaak_data/data_parth"
    output_data_dir = (
        "/home/parth/snaak/projects/granular_grasp/runs/train_w50_run_5/inference"
    )
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MassEstimationModel()
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model.eval()
    model.to(device)

    dataset = GraspDataset(
        create_transform_rgb(), create_transform_depth(), input_data_dir
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    run_inference_and_save_plots(
        model=model,
        data_loader=data_loader,
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

    # Load model
    model_path = "/home/parth/snaak/projects/granular_grasp/runs/train_w50_run_7/mass_estimation_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MassEstimationModel()
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model.eval()
    model.to(device)

    # Create transformation functions
    transform_rgb = create_transform_rgb()
    transform_depth = create_transform_depth()

    # Load image and depth map
    img_path = "/home/parth/snaak/projects/granular_grasp/rgb_image.jpg"
    img = cv2.imread(img_path)
    depth_map = np.load("/home/parth/snaak/projects/granular_grasp/depth_map.npy")

    # Run inference on the bin
    infer_on_bin(img, depth_map, model, transform_rgb, transform_depth, device)


def test_get_xy_for_weight():
    # Load model
    model_path = (
        "/home/parth/snaak/projects/granular_grasp/best_run/mass_estimation_model.pth"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MassEstimationModel()
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model.eval()
    model.to(device)

    # Create transformation functions
    transform_rgb = create_transform_rgb()
    transform_depth = create_transform_depth()

    # Load image and depth map
    img_path = "/home/parth/snaak/projects/granular_grasp/rgb_image.jpg"
    img = cv2.imread(img_path)
    depth_map = np.load("/home/parth/snaak/projects/granular_grasp/depth_map.npy")

    best_x, best_y = get_xy_for_weight(
        6, img, depth_map, model, device, transform_rgb, transform_depth
    )
    print(f"Best x: {best_x}, Best y: {best_y}")


def infer_on_extracted_data():
    desired_weight = 10.0
    rgb_save_dir = "/home/parth/snaak/snaak_data/extracted/rgb"
    depth_save_dir = "/home/parth/snaak/snaak_data/extracted/depth"
    output_save_dir = "/home/parth/snaak/snaak_data/extracted/inference"

    model_path = (
        "/home/parth/snaak/projects/granular_grasp/best_run/mass_estimation_model.pth"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MassEstimationModel()
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model.eval()
    model.to(device)

    # Create transformation functions
    transform_rgb = create_transform_rgb()
    transform_depth = create_transform_depth()

    rgb_files = sorted(os.listdir(rgb_save_dir))
    depth_files = sorted(os.listdir(depth_save_dir))

    for rgb_file, depth_file in tqdm(zip(rgb_files, depth_files), total=len(rgb_files)):
        rgb_path = os.path.join(rgb_save_dir, rgb_file)
        depth_path = os.path.join(depth_save_dir, depth_file)
        rgb = cv2.imread(rgb_path)
        depth = np.load(depth_path)
        save_path = os.path.join(output_save_dir, f"pred_{rgb_file}")
        best_x, best_y = get_xy_for_weight(
            desired_weight,
            rgb,
            depth,
            model,
            device,
            transform_rgb,
            transform_depth,
            save_path,
        )


if __name__ == "__main__":
    infer_on_extracted_data()
