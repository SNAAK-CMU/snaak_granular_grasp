import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as T

DEFAULT_DATA_DIR = "/home/parth/snaak/snaak_data/data_parth"
WINDOW_SIZE = 50  # (pixels) The model architecture depends on this!


####### Ensure these are same for the dataloader and the data collection node #######

# Bin dimensions
BIN_WIDTH_M = 0.140
BIN_LENGTH_M = 0.240
BIN_HEIGHT = 0.065
BIN_WIDTH_PIX = 189
BIN_LENGTH_PIX = 326
CAM2BIN_DIST_MM = 320

# For cropping the bin from rgb and depth image
CROP_XMIN = 274
CROP_XMAX = 463
CROP_YMIN = 0
CROP_YMAX = 326

#####################################################################################

# For filtering good grasps
GOOD_Z_BELOW_SURFACE = 0.03
# Reject grasps with weight outside this range
WEIGHT_MIN = 0.0
WEIGHT_MAX = 20.0


class CoordConverter:
    def __init__(
        self,
        bin_width_m=BIN_WIDTH_M,
        bin_length_m=BIN_LENGTH_M,
        bin_height_m=BIN_HEIGHT,
        bin_width_pix=BIN_WIDTH_PIX,
        bin_length_pix=BIN_LENGTH_PIX,
    ):
        self.bin_width_m = bin_width_m
        self.bin_length_m = bin_length_m
        self.bin_height_m = bin_height_m
        self.bin_width_pix = bin_width_pix
        self.bin_length_pix = bin_length_pix

    def m_to_pix(self, x_m, y_m):
        x_pix = int(x_m * self.bin_width_pix / self.bin_width_m)
        y_pix = int(y_m * self.bin_length_pix / self.bin_length_m)
        return x_pix, y_pix

    def pix_to_m(self, x_pix, y_pix):
        x_m = x_pix * self.bin_width_m / self.bin_width_pix
        y_m = y_pix * self.bin_length_m / self.bin_length_pix
        return x_m, y_m

    def action_xy_to_pix(self, action_x_m, action_y_m, img_w, img_h):
        x_disp_pix, y_disp_pix = self.m_to_pix(action_x_m, action_y_m)
        # print(f"x_disp_pix: {x_disp_pix}, y_disp_pix: {y_disp_pix}")
        action_x_pix = img_w // 2 - x_disp_pix
        action_y_pix = img_h // 2 + y_disp_pix

        # Clip the action points to the image boundaries
        action_x_pix = np.clip(action_x_pix, 0, img_w - 1)
        action_y_pix = np.clip(action_y_pix, 0, img_h - 1)

        # print("Img center x, y: ", img_w // 2, img_h // 2)
        # print(f"action_x_pix: {action_x_pix}, action_y_pix: {action_y_pix}")
        return (action_x_pix, action_y_pix)


def create_transform_rgb():
    # Step 1: Basic transformations
    transform_list = [
        # Convert PIL Image to tensor (automatically converts to float32 and scales to [0, 1])
        T.ToTensor(),
    ]

    transform = T.Compose(transform_list)
    return transform


def create_transform_depth():
    transform_list = [
        T.ToTensor(),
        T.Lambda(lambda x: x - CAM2BIN_DIST_MM),
        T.Lambda(
            lambda x: torch.where(x < 0, torch.tensor(BIN_HEIGHT * 1000), x)
        ),  # Replace negative values with depth to bin bottom
    ]
    transform = T.Compose(transform_list)
    return transform


class GraspDataset(Dataset):
    def __init__(self, transform_rgb, transform_depth, data_dir=DEFAULT_DATA_DIR):
        super().__init__()
        self.data_dir = data_dir
        self.coord_converter = CoordConverter()
        self.transform_rgb = transform_rgb
        self.transform_depth = transform_depth

        # Extract data from the npz files
        self.rgb_images = []
        self.depth_images = []
        self.weight_labels = []
        # self.start_weights = []
        # self.final_weights = []
        self.z_below_surface = []
        # self.actions = []

        self.npz_files = glob.glob(
            os.path.join(self.data_dir, "**/*.npz"), recursive=True
        )
        for npz_file in self.npz_files:
            data = np.load(npz_file, allow_pickle=True)
            # self.rgb_images.append(data["rgb"])
            # self.depth_images.append(data["depth"])
            # self.start_weights.append(data["start_weight"])
            # self.final_weights.append(data["final_weight"])
            # self.actions.append(data["a1"])
            self.z_below_surface.append(data["z_below_surface"])
            rgb_cropped, depth_cropped = self.__crop_rgbd_bin(
                data["rgb"], data["depth"]
            )
            rgb_patch, depth_patch = self.__crop_rgbd_patch(
                data["a1"], rgb_cropped, depth_cropped
            )
            self.rgb_images.append(rgb_patch)
            self.depth_images.append(depth_patch)
            self.weight_labels.append(data["start_weight"] - data["final_weight"])

        # Only keep the good grasps
        self.rgb_images, self.depth_images, self.weight_labels = (
            self.__filter_good_grasps(
                self.rgb_images,
                self.depth_images,
                self.weight_labels,
                self.z_below_surface,
            )
        )

    def __filter_good_grasps(
        self, rgb_images, depth_images, weight_labels, z_below_surface
    ):
        good_rgb_images = []
        good_depth_images = []
        good_weight_labels = []

        for rgb, depth, weight_label, z_below_surface in zip(
            rgb_images, depth_images, weight_labels, z_below_surface
        ):
            if (z_below_surface == GOOD_Z_BELOW_SURFACE) and (
                WEIGHT_MIN <= weight_label <= WEIGHT_MAX
            ):
                good_rgb_images.append(rgb)
                good_depth_images.append(depth)
                good_weight_labels.append(weight_label)
        return good_rgb_images, good_depth_images, good_weight_labels

    def __crop_rgbd_bin(self, rgb, depth):
        cropped_rgb = rgb[CROP_YMIN:CROP_YMAX, CROP_XMIN:CROP_XMAX, :]
        cropped_depth = depth[CROP_YMIN:CROP_YMAX, CROP_XMIN:CROP_XMAX]
        return cropped_rgb, cropped_depth

    def __crop_rgbd_patch(self, action, rgb, depth):
        img_h, img_w, _ = rgb.shape

        # Pad the rgb image and the depth image with zeros
        pad = WINDOW_SIZE // 2
        rgb_padded = np.pad(
            rgb,
            ((pad, pad), (pad, pad), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        depth_padded = np.pad(
            depth,
            ((pad, pad), (pad, pad)),
            mode="constant",
            constant_values=CAM2BIN_DIST_MM,
        )

        action_x_pix, action_y_pix = self.coord_converter.action_xy_to_pix(
            action[0], action[1], img_w, img_h
        )
        # Adjust the action point for the padded image
        action_x_pix += pad
        action_y_pix += pad

        # Crop the rgb and depth patches
        crop_xmin = action_x_pix - (WINDOW_SIZE // 2)
        crop_xmax = crop_xmin + WINDOW_SIZE
        crop_ymin = action_y_pix - (WINDOW_SIZE // 2)
        crop_ymax = crop_ymin + WINDOW_SIZE

        DEBUG_PLOT = False  # DO NOT SET TO TRUE FOR PRODUCTION
        if DEBUG_PLOT:
            rgb_copy = rgb_padded.copy()
            depth_copy = depth_padded.copy()
            cv2.rectangle(
                rgb_copy, (crop_xmin, crop_ymin), (crop_xmax, crop_ymax), (0, 0, 255), 2
            )
            cv2.rectangle(
                depth_copy,
                (crop_xmin, crop_ymin),
                (crop_xmax, crop_ymax),
                (0, 0, 255),
                2,
            )
            cv2.circle(rgb_copy, (action_x_pix, action_y_pix), 5, (0, 0, 255), -1)
            cv2.circle(depth_copy, (action_x_pix, action_y_pix), 5, (0, 0, 255), -1)
            cv2.imshow("rgb_patch_plot", cv2.cvtColor(rgb_copy, cv2.COLOR_RGB2BGR))
            cv2.imshow("depth_patch_plot", depth_copy)

        rgb_patch = rgb_padded[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]
        depth_patch = depth_padded[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

        return rgb_patch, depth_patch

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb = self.rgb_images[idx]
        depth = self.depth_images[idx]
        weight_label = self.weight_labels[idx]

        # TODO: Normalize the rgb
        # TODO: Shift the depth to make the bin surface at 0

        rgb_patch = self.transform_rgb(rgb)
        depth_patch = self.transform_depth(depth)

        # Convert to torch tensor with proper data types
        rgb_patch = rgb_patch.to(torch.float32)
        depth_patch = depth_patch.to(torch.float32)
        weight_label = torch.tensor(weight_label.astype(np.float32))

        return (rgb_patch, depth_patch), weight_label


def test_dataset():
    dataset = GraspDataset()
    for i in range(len(dataset)):
        (rgb_patch, depth_patch), weight_label = dataset[i]
        print(f"Weight label: {weight_label}")

    # cv2.destroyAllWindows()


def main():
    from torch.utils.data import DataLoader

    dataset = GraspDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for i, ((rgb_patches, depth_patches), weight_labels) in enumerate(dataloader):
        # print(f"Batch {i}:")
        # print(f"  RGB patch shape: {rgb_patches.shape}")
        # print(f"  Depth patch shape: {depth_patches.shape}")
        # print(f"  Weight labels: {weight_labels}")

        # Convert tensors to numpy arrays for visualization
        rgb_np = rgb_patches[0].numpy()  # shape: (H, W, C) or (C, H, W)
        depth_np = depth_patches[0].numpy()
        weight = (
            weight_labels[0].item()
            if hasattr(weight_labels[0], "item")
            else float(weight_labels[0])
        )

        # If rgb is (C, H, W), transpose to (H, W, C)
        if rgb_np.shape[0] in [1, 3] and rgb_np.shape[0] != rgb_np.shape[-1]:
            rgb_np = np.transpose(rgb_np, (1, 2, 0))

        # Write the weight on the rgb image
        cv2.putText(
            rgb_np,
            f"Weight: {weight:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow(f"rgb_patch_{i}.jpg", cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
        cv2.imshow(f"depth_patch_{i}.png", depth_np)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        break


if __name__ == "__main__":
    test_dataset()
