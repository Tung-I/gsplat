import numpy as np
import cv2
import imageio
from scipy.spatial.transform import Rotation as R
import os
import os.path as osp
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Optional
import numpy
import skimage.io


def perform_dibr(
    source_color_image,
    depth,
    source_cam_to_world,
    target_cam_to_world,
    K
):
    """
    Perform Depth Image-Based Rendering (DIBR) to warp the source image to the target viewpoint.

    Args:
        source_color_image: (H, W, 3) numpy array of source color image.
        depth: (H, W) numpy array of source depth image.
        source_cam_to_world: (4, 4) numpy array of source camera-to-world matrix.
        target_cam_to_world: (4, 4) numpy array of target camera-to-world matrix.
        K: (3, 3) numpy array of camera intrinsics.

    Returns:
        warped_image: (H, W, 3) numpy array of the warped image.
    """
    height, width = depth.shape
    # Prepare pixel grid
    u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))
    u_coords = u_coords.flatten().astype(np.int32)
    v_coords = v_coords.flatten().astype(np.int32)
    ones = np.ones_like(u_coords)

    # Convert pixel coordinates to normalized device coordinates
    pixel_coords = np.stack([u_coords, v_coords, ones], axis=1).T  # Shape: (3, N)

    # Compute depths
    depths = depth.flatten()  # Shape: (N,)

    # Back-project pixels to 3D points in source camera frame
    K_inv = np.linalg.inv(K)
    camera_points = K_inv @ (pixel_coords * depths)

    # Convert points to homogeneous coordinates
    homogeneous_camera_points = np.vstack((camera_points, ones))

    # Transform points to world coordinates
    world_points = source_cam_to_world @ homogeneous_camera_points  # Shape: (4, N)

    # Transform points to target camera coordinates
    target_world_to_cam = np.linalg.inv(target_cam_to_world)
    target_camera_points_homogeneous = target_world_to_cam @ world_points  # Shape: (4, N)
    target_camera_points = target_camera_points_homogeneous[:3, :]  # Exclude homogeneous coordinate

    # Project points onto target image plane
    projected_pixels = K @ target_camera_points
    projected_pixels /= projected_pixels[2, :]  # Normalize by z
    u_projected = projected_pixels[0, :]
    v_projected = projected_pixels[1, :]
    z_projected = target_camera_points[2, :]

    # Filter out points that are behind the camera (z <= 0)
    valid_depths = z_projected > 0
    u_projected = u_projected[valid_depths]
    v_projected = v_projected[valid_depths]
    z_projected = z_projected[valid_depths]
    u_source = u_coords[valid_depths]
    v_source = v_coords[valid_depths]

    # Round projected pixel coordinates to nearest integer
    u_projected_int = np.round(u_projected).astype(int)
    v_projected_int = np.round(v_projected).astype(int)

    # Filter out pixels that are outside the image boundaries
    height, width, _ = source_color_image.shape
    valid_pixels = (
        (u_projected_int >= 0) & (u_projected_int < width) &
        (v_projected_int >= 0) & (v_projected_int < height)
    )
    u_projected_int = u_projected_int[valid_pixels]
    v_projected_int = v_projected_int[valid_pixels]
    z_projected = z_projected[valid_pixels]
    u_source = u_source[valid_pixels]
    v_source = v_source[valid_pixels]

    # Prepare the depth buffer and warped image
    depth_buffer = np.full((height, width), np.inf)
    warped_image = np.zeros_like(source_color_image)

    # Get source colors
    source_colors = source_color_image[v_source, u_source, :]  # Shape: (N_valid, 3)

    # Flatten indices for depth buffer and warped image
    target_indices = v_projected_int * width + u_projected_int

    # For handling multiple points mapping to the same pixel, we use Z-buffering
    # We need to process the points in order of increasing depth (closest points first)
    sort_indices = np.argsort(z_projected)
    target_indices_sorted = target_indices[sort_indices]
    z_projected_sorted = z_projected[sort_indices]
    source_colors_sorted = source_colors[sort_indices]

    # Update the warped image using the closest points
    for idx in range(len(target_indices_sorted)):
        pix_idx = target_indices_sorted[idx]
        y = pix_idx // width
        x = pix_idx % width
        z = z_projected_sorted[idx]
        color = source_colors_sorted[idx]

        if depth_buffer[y, x] > z:
            depth_buffer[y, x] = z
            warped_image[y, x, :] = color

    return warped_image


class Warper:
    def __init__(self, resolution: tuple = None):
        self.resolution = resolution
        return

    def forward_warp(self, frame1: numpy.ndarray, mask1: Optional[numpy.ndarray], depth1: numpy.ndarray,
                     transformation1: numpy.ndarray, transformation2: numpy.ndarray, intrinsic1: numpy.ndarray,
                     intrinsic2: Optional[numpy.ndarray]) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray,
                                                                   numpy.ndarray]:
        """
        Given a frame1 and global transformations transformation1 and transformation2, warps frame1 to next view using
        bilinear splatting.
        :param frame1: (h, w, 3) uint8 numpy array
        :param mask1: (h, w) bool numpy array. Wherever mask1 is False, those pixels are ignored while warping. Optional
        :param depth1: (h, w) float numpy array.
        :param transformation1: (4, 4) extrinsic transformation matrix of first view: [R, t; 0, 1]
        :param transformation2: (4, 4) extrinsic transformation matrix of second view: [R, t; 0, 1]
        :param intrinsic1: (3, 3) camera intrinsic matrix
        :param intrinsic2: (3, 3) camera intrinsic matrix. Optional
        """
        if self.resolution is not None:
            assert frame1.shape[:2] == self.resolution
        h, w = frame1.shape[:2]
        if mask1 is None:
            mask1 = numpy.ones(shape=(h, w), dtype=bool)
        if intrinsic2 is None:
            intrinsic2 = numpy.copy(intrinsic1)
        assert frame1.shape == (h, w, 3)
        assert mask1.shape == (h, w)
        assert depth1.shape == (h, w)
        print(f"Depth max value: {depth1.max()}, min value: {depth1.min()}")
        assert transformation1.shape == (4, 4)
        assert transformation2.shape == (4, 4)
        assert intrinsic1.shape == (3, 3)
        assert intrinsic2.shape == (3, 3)

        trans_points1 = self.compute_transformed_points(depth1, transformation1, transformation2, intrinsic1,
                                                        intrinsic2)
        # perspective division
        trans_coordinates = trans_points1[:, :, :2, 0] / trans_points1[:, :, 2:3, 0]
        trans_depth1 = trans_points1[:, :, 2, 0]

        grid = self.create_grid(h, w)
        flow12 = trans_coordinates - grid

        warped_frame2, mask2 = self.bilinear_splatting(frame1, mask1, trans_depth1, flow12, None, is_image=True)
        warped_depth2 = self.bilinear_splatting(trans_depth1[:, :, None], mask1, trans_depth1, flow12, None,
                                                is_image=False)[0][:, :, 0]
        return warped_frame2, mask2, warped_depth2, flow12

    def compute_transformed_points(self, depth1: numpy.ndarray, transformation1: numpy.ndarray,
                                   transformation2: numpy.ndarray, intrinsic1: numpy.ndarray,
                                   intrinsic2: Optional[numpy.ndarray]):
        """
        Computes transformed position for each pixel location
        """
        if self.resolution is not None:
            assert depth1.shape == self.resolution
        h, w = depth1.shape
        if intrinsic2 is None:
            intrinsic2 = numpy.copy(intrinsic1)
        transformation = numpy.matmul(transformation2, numpy.linalg.inv(transformation1))

        y1d = numpy.array(range(h))
        x1d = numpy.array(range(w))
        x2d, y2d = numpy.meshgrid(x1d, y1d)
        ones_2d = numpy.ones(shape=(h, w))
        ones_4d = ones_2d[:, :, None, None]
        pos_vectors_homo = numpy.stack([x2d, y2d, ones_2d], axis=2)[:, :, :, None]  # (h, w, 3, 1)

        intrinsic1_inv = numpy.linalg.inv(intrinsic1)
        intrinsic1_inv_4d = intrinsic1_inv[None, None]
        intrinsic2_4d = intrinsic2[None, None]  # (1, 1, 3, 3)
        depth_4d = depth1[:, :, None, None]  # (h, w, 1, 1)
        trans_4d = transformation[None, None]  # (1, 1, 4, 4)

        unnormalized_pos = numpy.matmul(intrinsic1_inv_4d, pos_vectors_homo)  # (h, w, 3, 1)
        world_points = depth_4d * unnormalized_pos
        world_points_homo = numpy.concatenate([world_points, ones_4d], axis=2) # (h, w, 4, 1)
        trans_world_homo = numpy.matmul(trans_4d, world_points_homo) # (h, w, 4, 1)
        trans_world = trans_world_homo[:, :, :3]
        trans_norm_points = numpy.matmul(intrinsic2_4d, trans_world) # (h, w, 3, 1)
        # the projected points are in homo coordinates, still need perspective division 
        return trans_norm_points

    def bilinear_splatting(self, frame1: numpy.ndarray, mask1: Optional[numpy.ndarray], depth1: numpy.ndarray,
                           flow12: numpy.ndarray, flow12_mask: Optional[numpy.ndarray], is_image: bool = False) -> \
            Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Using inverse bilinear interpolation based splatting
        :param frame1: (h, w, c)
        :param mask1: (h, w): True if known and False if unknown. Optional
        :param depth1: (h, w)
        :param flow12: (h, w, 2)
        :param flow12_mask: (h, w): True if valid and False if invalid. Optional
        :param is_image: If true, the return array will be clipped to be in the range [0, 255] and type-casted to uint8
        :return: warped_frame2: (h, w, c)
                 mask2: (h, w): True if known and False if unknown
        """
        if self.resolution is not None:
            assert frame1.shape[:2] == self.resolution
        h, w, c = frame1.shape
        if mask1 is None:
            mask1 = numpy.ones(shape=(h, w), dtype=bool)
        if flow12_mask is None:
            flow12_mask = numpy.ones(shape=(h, w), dtype=bool)
        grid = self.create_grid(h, w)
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1  # account for padding
        trans_pos_floor = numpy.floor(trans_pos_offset).astype('int')
        trans_pos_ceil = numpy.ceil(trans_pos_offset).astype('int')
        # Ensure that the indices are within the image bounds
        trans_pos_offset[:, :, 0] = numpy.clip(trans_pos_offset[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_offset[:, :, 1] = numpy.clip(trans_pos_offset[:, :, 1], a_min=0, a_max=h + 1)
        trans_pos_floor[:, :, 0] = numpy.clip(trans_pos_floor[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_floor[:, :, 1] = numpy.clip(trans_pos_floor[:, :, 1], a_min=0, a_max=h + 1)
        trans_pos_ceil[:, :, 0] = numpy.clip(trans_pos_ceil[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_ceil[:, :, 1] = numpy.clip(trans_pos_ceil[:, :, 1], a_min=0, a_max=h + 1)

        # The weight is the product of the distances in the x and y directions
        prox_weight_nw = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                         (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                         (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
        prox_weight_ne = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                         (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))
        prox_weight_se = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                         (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))

        # Computing Depth Weights
        sat_depth1 = numpy.clip(depth1, a_min=0, a_max=1000)
        log_depth1 = numpy.log(1 + sat_depth1)
        depth_weights = numpy.exp(log_depth1 / log_depth1.max() * 50)

        # Combining Weights
        weight_nw = prox_weight_nw * mask1 * flow12_mask / depth_weights
        weight_sw = prox_weight_sw * mask1 * flow12_mask / depth_weights
        weight_ne = prox_weight_ne * mask1 * flow12_mask / depth_weights
        weight_se = prox_weight_se * mask1 * flow12_mask / depth_weights

        weight_nw_3d = weight_nw[:, :, None]
        weight_sw_3d = weight_sw[:, :, None]
        weight_ne_3d = weight_ne[:, :, None]
        weight_se_3d = weight_se[:, :, None]

        warped_image = numpy.zeros(shape=(h + 2, w + 2, c), dtype=numpy.float64)
        warped_weights = numpy.zeros(shape=(h + 2, w + 2), dtype=numpy.float64)

        # For each pixel in the source image, its value (multiplied by the weight) is added to the four neighboring pixels in the target image.
        numpy.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), frame1 * weight_nw_3d)
        numpy.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), frame1 * weight_sw_3d)
        numpy.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), frame1 * weight_ne_3d)
        numpy.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), frame1 * weight_se_3d)

        numpy.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), weight_nw)
        numpy.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), weight_sw)
        numpy.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), weight_ne)
        numpy.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), weight_se)

        cropped_warped_image = warped_image[1:-1, 1:-1]
        cropped_weights = warped_weights[1:-1, 1:-1]

        mask = cropped_weights > 0
        with numpy.errstate(invalid='ignore'):
            warped_frame2 = numpy.where(mask[:, :, None], cropped_warped_image / cropped_weights[:, :, None], 0)

        if is_image:
            assert numpy.min(warped_frame2) >= 0
            assert numpy.max(warped_frame2) <= 256
            clipped_image = numpy.clip(warped_frame2, a_min=0, a_max=255)
            warped_frame2 = numpy.round(clipped_image).astype('uint8')
        return warped_frame2, mask

    def bilinear_interpolation(self, frame2: numpy.ndarray, mask2: Optional[numpy.ndarray], flow12: numpy.ndarray,
                               flow12_mask: Optional[numpy.ndarray], is_image: bool = False) -> \
            Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Using bilinear interpolation
        :param frame2: (h, w, c)
        :param mask2: (h, w): True if known and False if unknown. Optional
        :param flow12: (h, w, 2)
        :param flow12_mask: (h, w): True if valid and False if invalid. Optional
        :param is_image: If true, the return array will be clipped to be in the range [0, 255] and type-casted to uint8
        :return: warped_frame1: (h, w, c)
                 mask1: (h, w): True if known and False if unknown
        """
        if self.resolution is not None:
            assert frame2.shape[:2] == self.resolution
        h, w, c = frame2.shape
        if mask2 is None:
            mask2 = numpy.ones(shape=(h, w), dtype=bool)
        if flow12_mask is None:
            flow12_mask = numpy.ones(shape=(h, w), dtype=bool)
        grid = self.create_grid(h, w)
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = numpy.floor(trans_pos_offset).astype('int')
        trans_pos_ceil = numpy.ceil(trans_pos_offset).astype('int')
        trans_pos_offset[:, :, 0] = numpy.clip(trans_pos_offset[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_offset[:, :, 1] = numpy.clip(trans_pos_offset[:, :, 1], a_min=0, a_max=h + 1)
        trans_pos_floor[:, :, 0] = numpy.clip(trans_pos_floor[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_floor[:, :, 1] = numpy.clip(trans_pos_floor[:, :, 1], a_min=0, a_max=h + 1)
        trans_pos_ceil[:, :, 0] = numpy.clip(trans_pos_ceil[:, :, 0], a_min=0, a_max=w + 1)
        trans_pos_ceil[:, :, 1] = numpy.clip(trans_pos_ceil[:, :, 1], a_min=0, a_max=h + 1)

        prox_weight_nw = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                         (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                         (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
        prox_weight_ne = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                         (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))
        prox_weight_se = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                         (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))

        weight_nw = prox_weight_nw * flow12_mask
        weight_sw = prox_weight_sw * flow12_mask
        weight_ne = prox_weight_ne * flow12_mask
        weight_se = prox_weight_se * flow12_mask

        weight_nw_3d = weight_nw[:, :, None]
        weight_sw_3d = weight_sw[:, :, None]
        weight_ne_3d = weight_ne[:, :, None]
        weight_se_3d = weight_se[:, :, None]

        frame2_offset = numpy.pad(frame2, pad_width=((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
        mask2_offset = numpy.pad(mask2, pad_width=((1, 1), (1, 1)), mode='constant', constant_values=0)

        f2_nw = frame2_offset[trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]]
        f2_sw = frame2_offset[trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]]
        f2_ne = frame2_offset[trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]]
        f2_se = frame2_offset[trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]]

        m2_nw = mask2_offset[trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]]
        m2_sw = mask2_offset[trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]]
        m2_ne = mask2_offset[trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]]
        m2_se = mask2_offset[trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]]

        m2_nw_3d = m2_nw[:, :, None]
        m2_sw_3d = m2_sw[:, :, None]
        m2_ne_3d = m2_ne[:, :, None]
        m2_se_3d = m2_se[:, :, None]

        nr = weight_nw_3d * f2_nw * m2_nw_3d + weight_sw_3d * f2_sw * m2_sw_3d + \
             weight_ne_3d * f2_ne * m2_ne_3d + weight_se_3d * f2_se * m2_se_3d
        dr = weight_nw_3d * m2_nw_3d + weight_sw_3d * m2_sw_3d + weight_ne_3d * m2_ne_3d + weight_se_3d * m2_se_3d
        warped_frame1 = numpy.where(dr > 0, nr / dr, 0)
        mask1 = dr[:, :, 0] > 0

        if is_image:
            assert numpy.min(warped_frame1) >= 0
            assert numpy.max(warped_frame1) <= 256
            clipped_image = numpy.clip(warped_frame1, a_min=0, a_max=255)
            warped_frame1 = numpy.round(clipped_image).astype('uint8')
        return warped_frame1, mask1

    @staticmethod
    def create_grid(h, w):
        x_1d = numpy.arange(0, w)[None]
        y_1d = numpy.arange(0, h)[:, None]
        x_2d = numpy.repeat(x_1d, repeats=h, axis=0)
        y_2d = numpy.repeat(y_1d, repeats=w, axis=1)
        grid = numpy.stack([x_2d, y_2d], axis=2)
        return grid

    @staticmethod
    def read_image(path: Path) -> numpy.ndarray:
        if path.suffix in ['.jpg', '.png', '.bmp']:
            image = skimage.io.imread(path.as_posix())
        elif path.suffix == '.npy':
            image = numpy.load(path.as_posix())
        else:
            raise RuntimeError(f'Unknown image format: {path.as_posix()}')
        return image

    @staticmethod
    def read_depth(path: Path) -> numpy.ndarray:
        if path.suffix == '.png':
            depth = skimage.io.imread(path.as_posix())
        elif path.suffix == '.npy':
            depth = numpy.load(path.as_posix())
        elif path.suffix == '.npz':
            with numpy.load(path.as_posix()) as depth_data:
                depth = depth_data['depth']
        elif path.suffix == '.exr':
            import Imath
            import OpenEXR

            exr_file = OpenEXR.InputFile(path.as_posix())
            raw_bytes = exr_file.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
            depth_vector = numpy.frombuffer(raw_bytes, dtype=numpy.float32)
            height = exr_file.header()['displayWindow'].max.y + 1 - exr_file.header()['displayWindow'].min.y
            width = exr_file.header()['displayWindow'].max.x + 1 - exr_file.header()['displayWindow'].min.x
            depth = numpy.reshape(depth_vector, (height, width))
        else:
            raise RuntimeError(f'Unknown depth format: {path.as_posix()}')
        return depth

    @staticmethod
    def camera_intrinsic_transform(capture_width=1920, capture_height=1080, patch_start_point: tuple = (0, 0)):
        start_y, start_x = patch_start_point
        camera_intrinsics = numpy.eye(3)
        camera_intrinsics[0, 0] = 2100
        camera_intrinsics[0, 2] = capture_width / 2.0 - start_x
        camera_intrinsics[1, 1] = 2100
        camera_intrinsics[1, 2] = capture_height / 2.0 - start_y
        return camera_intrinsics

def main():
    # Paths to your data
    color_img_path = 'results/garden/renders/color_0010.png'
    depth_path = 'results/garden/renders/depth_0010.npy'

    # Load camera intrinsics K
    K = np.array([[480.612, 0.0, 324.1875],
                  [0.0, 481.5445, 210.0625],
                  [0.0, 0.0, 1.0]])

    # Load images
    color_image = imageio.imread(color_img_path)
    if color_image.dtype != np.uint8:
            color_image = (color_image * 255).astype(np.uint8)
    H, W = color_image.shape[:2]

    depth = np.load(depth_path)
    assert color_image.shape[:2] == depth.shape[:2]
    # Load camera-to-world matrices for source and target
    src_camtoworlds = np.load('results/garden/renders/camtoworlds_0010.npy')[0]
    tgt_camtoworlds = np.load('results/garden/renders/camtoworlds_0005.npy')[0]


    warped_image = perform_dibr(
            source_color_image=color_image,
            depth=depth,
            source_cam_to_world=src_camtoworlds,
            target_cam_to_world=tgt_camtoworlds,
            K=K
        )
    imageio.imwrite('results/garden/renders/warped_0010_0005.png', warped_image)

def demo1():
    # Paths to your data
    color_img_path = Path('results/garden/renders/color_0010.png')
    depth_path = Path('results/garden/renders/depth_0010.npy')

    warper = Warper()
    frame1 = warper.read_image(color_img_path)
    depth1 = warper.read_depth(depth_path )
    intrinsic = np.array([[480.612, 0.0, 324.1875],
                  [0.0, 481.5445, 210.0625],
                  [0.0, 0.0, 1.0]])
    src_camtoworlds = np.load('results/garden/renders/camtoworlds_0010.npy')[0]
    tgt_camtoworlds = np.load('results/garden/renders/camtoworlds_0005.npy')[0]
    inv_src_camtoworlds = np.linalg.inv(src_camtoworlds)
    inv_tgt_camtoworlds = np.linalg.inv(tgt_camtoworlds)

    warped_frame2 = warper.forward_warp(frame1, None, depth1, inv_src_camtoworlds, inv_tgt_camtoworlds, intrinsic, None)[0]
    skimage.io.imsave('results/garden/renders/warped.png', warped_frame2)
    return


if __name__ == '__main__':
    # main()
    demo1()