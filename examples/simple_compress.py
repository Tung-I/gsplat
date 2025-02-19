# Code for rendering SpaceTimeGaussians

import argparse
import json
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict
import os
import imageio
import numpy as np
import torch
from tqdm import trange

from datasets.colmap import Parser
from datasets.traj import (
    generate_interpolated_path,
    generate_ellipse_path_z,
    generate_spiral_path,
)

from gsplat.custom_rendering import rasterize_gaussian_images, rasterize_pixels
from utils import knn, rgb_to_sh
from sptgs import GaussianModel 
from tile_processing import group_gaussian_attributes_by_tiles

@dataclass
class Config:
    data_dir: str
    data_factor: int
    normalize_world_space: bool
    test_every: int
    global_scale: float
    camera_model: str
    result_dir: str
    render_traj_path: str
    sh_degree: int
    near_plane: float
    far_plane: float

def load_config_json(path: str) -> Config:
    with open(path, "r") as f:
        data = json.load(f)
    return Config(**data)

def create_splats(
    parser: Parser,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    feature_dim = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
):

    points = torch.from_numpy(parser.points).float()
    rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    return splats

def attr2img(feats, height=512, width=512, bitdepth=8, device="cuda"):
    """Given a shape [N, D] array, produce a quantized image of shape [height, width, D<=4].
    D=4 => store as RGBA, D=3 => store as RGB, D=1 => store as R or replicate as R=G=B if needed.
    Args:
        feats: [N, D] array
    Return:
        quant_img: [height, width, D<=4] array
        (mins, maxs, (height, width)): tuple of mins, maxs and image shape
    """
    device = feats.device
    if len(feats.shape) == 1:
        feats = feats.unsqueeze(-1)
    N, D = feats.shape
    total_pixels = height * width
    if total_pixels < N:
        raise ValueError(f"Not enough pixels {total_pixels} < N {N}.")

    # Padding
    pad = total_pixels - N
    if pad > 0:
        pad_data = torch.zeros((pad, D), dtype=feats.dtype, device=device)
        array_padded = torch.cat([feats, pad_data], dim=0)  # shape [total_pixels, D]
    else:
        array_padded = feats
    offset = N  

    # Quantization
    mins = torch.min(array_padded, dim=0)[0] 
    maxs = torch.max(array_padded, dim=0)[0]  
    max_val = (1 << bitdepth) - 1  # e.g. 255 for bitdepth=8
    quant_tensor = torch.zeros_like(array_padded, dtype=torch.float32, device=device)

    for d in range(D):
        mn = mins[d]
        mx = maxs[d]
        if mx.item() == mn.item():  # store 0 or some constant
            quant_tensor[:, d] = 0.0
        else:
            arr_d = array_padded[:, d]
            arr_norm = (arr_d - mn) / (mx - mn)
            arr_q = arr_norm * max_val
            arr_q = torch.round(arr_q)
            arr_q = torch.clamp(arr_q, 0, max_val)
            quant_tensor[:, d] = arr_q

    if bitdepth == 8:
        quant_tensor = quant_tensor.to(torch.uint8)
    elif bitdepth == 16:
        quant_tensor = quant_tensor.to(torch.uint16)
    else:
        raise ValueError(f"Unsupported bitdepth {bitdepth}.")
    quant_tensor = quant_tensor.reshape(height, width, D)

    if D == 1:
        quant_tensor = quant_tensor.repeat(1, 1, 3)  # shape [H, W, 3]


    return quant_tensor, (mins, maxs, offset, D)


def img2attr(quant_img, meta, bitdepth=8):
    """Inverse transform from a quantized tensor to the original attributes [N, D].
    Args:
      quant_img: [height, width, D] (uint8)
      meta: (mins, maxs, offset, D_old) from attr2img_torch
      bitdepth: e.g. 8

    Returns:
      recons_feat: shape [N, D] float32 
    """
    device = quant_img.device
    (mins, maxs, offset, D_old) = meta
    if D_old==1:
        flatten = quant_img[..., 0].reshape(-1, 1)  # [total_pixels, 1]
    else:
        flatten = quant_img.reshape(-1, quant_img.shape[2])  # [total_pixels, D]
    flatten = flatten[:offset, :]
    flatten = flatten.float()
    max_val = (1 << bitdepth) - 1  # e.g. 255 for 8-bit
    D = flatten.shape[1]
    recons = torch.zeros_like(flatten, dtype=torch.float32, device=device)

    for d in range(D):
        mn = mins[d]
        mx = maxs[d]
        if mx.item() == mn.item():
            recons[:, d] = mn
        else:
            recons[:, d] = flatten[:, d] / max_val * (mx - mn) + mn

    if D==1:
        recons = recons.squeeze(-1)

    return recons 


def quantize_attribute(
    feats: torch.Tensor,   # shape [N, D], float32
    bitdepth: int = 8
):
    device = feats.device
    if len(feats.shape) == 1:
        feats = feats.unsqueeze(-1)
    N, D = feats.shape
    mins = torch.min(feats, dim=0)[0]  # shape [D]
    maxs = torch.max(feats, dim=0)[0]  # shape [D]

    max_val = (1 << bitdepth) - 1

    quant_tensor = torch.zeros_like(feats, dtype=torch.float32, device=device)

    # For each dimension, do min->max normalization
    for d in range(D):
        mn = mins[d]
        mx = maxs[d]
        if mx.item() == mn.item():
            quant_tensor[:, d] = 0.0
        else:
            arr_d = feats[:, d]
            arr_norm = (arr_d - mn) / (mx - mn)      # [0..1]
            arr_q = arr_norm * max_val              # [0..255 or 65535]
            arr_q = torch.round(arr_q)
            arr_q = torch.clamp(arr_q, 0, max_val)
            quant_tensor[:, d] = arr_q

    if bitdepth == 8:
        quant_tensor = quant_tensor.to(torch.uint8)
    elif bitdepth == 16:
        quant_tensor = quant_tensor.to(torch.uint16)
    else:
        raise ValueError(f"Unsupported bitdepth {bitdepth}.")

    return quant_tensor, (mins, maxs)


def dequantize_attribute(
    quant_tensor, minmax, bitdepth: int = 8
):
    """
    Dequantize a [N, D] tensor from uint8/uint16 back to float32, given stored mins & maxs.
    Args:
      quant_tensor: [N, D], dtype=uint8 or uint16
      minmax: (mins, maxs) each shape [D], from quantize_attribute
      bitdepth: 8 or 16
    Returns:
      recons: shape [N, D], float32
    """
    device = quant_tensor.device
    (mins, maxs) = minmax
    if len(quant_tensor.shape) == 1:
        quant_tensor = quant_tensor.unsqueeze(-1)
    N, D = quant_tensor.shape
    max_val = (1 << bitdepth) - 1

    flatten = quant_tensor.float()  # [N, D]

    recons = torch.zeros_like(flatten, dtype=torch.float32, device=device)
    for d in range(D):
        mn = mins[d]
        mx = maxs[d]
        if mx.item() == mn.item():
            recons[:, d] = mn
        else:
            recons[:, d] = flatten[:, d] / max_val * (mx - mn) + mn

    if recons.shape[1]==1:
        recons = recons.squeeze(-1)

    return recons

def pack_3dgs_into_single_image(
    means16:  torch.Tensor,      # shape [N,3], dtype=uint16, range [0..65535]
    quats:   torch.Tensor,       # shape [N, 4], 8-bit 
    scales:  torch.Tensor,       # shape [N, 3], 8-bit
    colors:  torch.Tensor,       # shape [N, 3], 8-bit
    opacities: torch.Tensor,     # shape [N],    8-bit
    height=720,
    width=1080,
):
    """
    Produces one single 3-channel image [720,1080,3], containing 6 tiles, each tile is
    360 x 360 x 3:

      tile(0,0) => means16_low 
      tile(0,1) => means16_high
      tile(0,2) => quats[:,:3]
      tile(1,0) => [quat[:,3], opacity, opacity]
      tile(1,1) => scales
      tile(1,2) => colors

    All must be in 8-bit format (uint8).
    We assume 'means16' is already in [N,3] range [0..65535].
    We'll first split means16 into two 8-bit arrays.

    quats, scales, colors, opacities => each is shape [N,(1..4)], dtype=uint8 => already quantized.

    Return:
      final_img: shape [720,1080,3], dtype=uint8
    """

    device = means16.device
    N = means16.shape[0]
    tileH = 360
    tileW = 360
    final_img = torch.zeros((height, width, 3), dtype=torch.uint8, device=device)

    def split_16bits_positions(positions_16b: torch.Tensor):
        """
        positions_16b: shape [N, 3], each element in [0..65535], dtype=torch.uint16 (on GPU).
        Returns: (low_bytes, high_bytes) each shape [N, 3], dtype=torch.uint8.
        
        Implementation detail: Because PyTorch does not support bitwise_and for uint16
        on GPU, we cast to int32 first.
        """
        # positions_16b = (N,3), uint16
        # cast to int32 so we can do bitwise operations
        pos_i32 = positions_16b.to(torch.int32)

        low  = (pos_i32 & 0x00ff).to(torch.uint8)               # lower 8 bits
        high = ((pos_i32 >> 8) & 0x00ff).to(torch.uint8)        # higher 8 bits
        return low, high

    def fill_tile(img, tile_idx_r, tile_idx_c, data_2d):
        """Fill a tile of the final image.
        Args:
            img: [height, width, 3] (uint8)
            tile_idx_r: int, row index of the tile
            tile_idx_c: int, column index of the tile
            data_2d: [N, 3] tensor
        """
        row0 = tile_idx_r * tileH
        col0 = tile_idx_c * tileW

        needed = tileH*tileW - data_2d.shape[0]
        if needed<0:
            raise ValueError("Not enough space in tile!")
        if needed>0:
            pad_data = torch.zeros((needed, 3), dtype=torch.uint8, device=device)
            data_2d = torch.cat([data_2d, pad_data], dim=0)

        # reshape => [tileH, tileW, 3]
        data_2d = data_2d.reshape(tileH, tileW, 3)
        img[row0:row0+tileH, col0:col0+tileW, :] = data_2d

    # 1) split means16 => means16_low, means16_high
    means16_low, means16_high = split_16bits_positions(means16)  # each [N,3], uint8
    fill_tile(final_img, 0, 0, means16_low)
    fill_tile(final_img, 0, 1, means16_high)

    quats_3 = quats[:, :3]  # shape [N,3], already uint8
    fill_tile(final_img, 0, 2, quats_3)

    # 4) tile3 => the 4th quat dimension + opacities => shape [N, 2]? 
    if opacities.dim()==1:
        opacities = opacities.unsqueeze(1)  # shape [N,1]
    r_chan = quats[:, 3:4]      # shape [N,1]
    g_chan = opacities  # replicate opacities into 2 channels
    b_chan = opacities
    tile3_data = torch.cat([r_chan, g_chan, b_chan], dim=1)  # [N,3]
    fill_tile(final_img, 1, 0, tile3_data)

    fill_tile(final_img, 1, 1, scales)
    fill_tile(final_img, 1, 2, colors)

    return final_img

def unpack_3dgs_from_single_image(
    final_img: torch.Tensor,
    offset: int,
):
    """
    Reverse of 'pack_3dgs_into_single_image'.
    final_img: [720,1080,3], dtype=uint8, device=...
    N: number of valid points

    Returns:
      means16:   shape [N,3], dtype=uint16
      quats:     shape [N,4], dtype=uint8
      scales:    shape [N,3], dtype=uint8
      colors:    shape [N,3], dtype=uint8
      opacities: shape [N],   dtype=uint8
    """
    device = final_img.device
    tileH = 360
    tileW = 360

    def combine_16bits_positions(low_bytes: torch.Tensor, high_bytes: torch.Tensor) -> torch.Tensor:
        """
        Inverse of split_16bits_positions.
        low_bytes, high_bytes: each shape [N, 3], dtype=uint8
        returns positions_16b: shape [N, 3], dtype=uint16 in [0..65535].
        
        Implementation detail: we cast low/high to int32, do bitwise ops,
        clamp to [0..65535], then cast to uint16.
        """
        # shape => [N,3], each is uint8
        low_i32  = low_bytes.to(torch.int32)
        high_i32 = high_bytes.to(torch.int32)
        # combine
        pos_i32 = (high_i32 << 8) | low_i32    # shape [N,3]
        # ensure range [0..65535]
        pos_i32 = torch.clamp(pos_i32, min=0, max=65535)
        positions_16b = pos_i32.to(torch.uint16)
        return positions_16b
    
    def read_tile(img, tile_idx_r, tile_idx_c, N):
        row0 = tile_idx_r*tileH
        col0 = tile_idx_c*tileW
        tile_data = img[row0:row0+tileH, col0:col0+tileW, :]
        tile_data = tile_data.reshape(-1, tile_data.shape[2])
        tile_data = tile_data[:N, :]  # shape [N,3]
        return tile_data

    # read tiles
    means16_low  = read_tile(final_img, 0, 0, offset)  # [N,3], uint8
    means16_high = read_tile(final_img, 0, 1, offset)
    quats_3      = read_tile(final_img, 0, 2, offset)
    tile3_data   = read_tile(final_img, 1, 0, offset)  # shape [N,3]
    scales       = read_tile(final_img, 1, 1, offset)
    colors       = read_tile(final_img, 1, 2, offset)

    # decode tile3 => R= quats[:,3], G=B => opacities
    r_chan = tile3_data[:, 0]
    g_chan = tile3_data[:, 1]
    # We assume B is the same as G => skip it or average it
    quats4 = torch.cat([quats_3, r_chan.unsqueeze(1)], dim=1)  # [N,4], uint8
    opacities = g_chan  # shape [N], uint8

    # combine means16_low, means16_high => means16
    means16 = combine_16bits_positions(means16_low, means16_high)  # shape [N,3], uint16

    return means16, quats4, scales, colors, opacities


def save_packed_frame(gs_img, out_dir, frame_id):
    """
    gs_img: torch.Tensor, shape (H, W, 3), dtype=uint8
    out_dir: str, directory path to store the frames
    frame_id: int, used for naming
    """
    # Make sure directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Convert to a NumPy array if it's still a torch.Tensor
    # shape [H, W, 3], dtype=uint8
    frame_np = gs_img.cpu().numpy()  

    # Name the file in a zero-padded way so FFmpeg sees them as a sequence
    filename = os.path.join(out_dir, f"packed_{frame_id:04d}.png")
    imageio.imwrite(filename, frame_np)

def load_decoded_frame(frame_path, device="cuda"):
    """
    Reads a decoded PNG file from disk and returns a torch.Tensor shape [H, W, 3], dtype=uint8.
    """
    img = imageio.imread(frame_path)  # shape (H, W, 3), np.uint8
    tensor = torch.from_numpy(img)    # shape (H, W, 3), torch.uint8
    tensor = tensor.to(device)
    return tensor

class DynamicGSRenderer:
    def __init__(self, cfg: dict, ckpt: str, save_video: bool, save_frame: bool, save_first_frame: bool):
        self.cfg = cfg
        self.ckpt = ckpt
        self.save_video = save_video
        self.save_frame = save_frame
        self.save_first_frame = save_first_frame
        self.local_rank = 0
        self.world_rank = 0
        self.world_size = 1
        self.device='cuda'

        # Data parser for cameras
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.scene_scale = self.parser.scene_scale * cfg.global_scale
        print(f"Scene scale: {self.scene_scale:.4f}")

        # Create container for GS model
        self.splats = create_splats(
            self.parser,
            init_opacity=0.1,
            init_scale=1.0,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            feature_dim=None,
            device="cuda",
            world_rank=self.world_rank,
            world_size=self.world_size,
        )

        # Load SPTGS pretrained model
        self.model = GaussianModel(sh_degree=cfg.sh_degree)
        ply_file = os.path.join(ckpt, "point_cloud.ply")
        if not os.path.exists(ply_file):
            raise FileNotFoundError(f"Could not find {ply_file} for the dynamic model.")
        self.model.load_ply(ply_file)
        print(f"[SPTGSRenderer] Loaded SPTGS model with {self.model._xyz.shape[0]} points.")

        # Create output directories
        self.result_dir = cfg.result_dir
        os.makedirs(self.result_dir, exist_ok=True)
        self.video_dir = os.path.join(self.result_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)
        self.render_dir = os.path.join(self.result_dir, "renders")
        os.makedirs(self.render_dir, exist_ok=True)

    @torch.no_grad()
    def load_frame(self, splats, timestamp: float):
        """Parse the frame of SPTGS at a given timestamp.
        Args:
            splats: dict, splats to be updated
            timestamp: float
        Outputs:
            None
        """
        means3D = self.model.get_xyz
        pointopacity = self.model.get_opacity
        trbfunction = lambda x: torch.exp(-x.pow(2))
        trbf_center = self.model.get_trbfcenter
        trbf_scale = self.model.get_trbfscale
        N = means3D.shape[0]

        # compute offset
        pointtimes = torch.ones((N,1), dtype=means3D.dtype, requires_grad=False, device="cuda") + 0  
        trbfdistanceoffset = timestamp * pointtimes - trbf_center
        trbfdistance =  trbfdistanceoffset / torch.exp(trbf_scale) 
        trbfoutput = trbfunction(trbfdistance)
        self.model.trbfoutput = trbfoutput

        opacity = pointopacity * trbfoutput  # - 0.5
        opacity = opacity.squeeze(-1)

        tforpoly = trbfdistanceoffset.detach()
        means3D = means3D +  self.model._motion[:, 0:3] * tforpoly + \
            self.model._motion[:, 3:6] * tforpoly * tforpoly + \
                self.model._motion[:, 6:9] * tforpoly *tforpoly * tforpoly
        
        rotations = self.model.get_rotation(tforpoly)

        color_precomp = self.model.get_features()

        splats["means"] = means3D
        splats["quats"] = rotations
        splats["scales"] = self.model.get_scaling  # raw log-scale
        splats["opacities"] = opacity  # raw logit + time-based multiplier
        splats["colors"] = color_precomp
        return 

    @torch.no_grad()
    def rasterize_splats(
        self,
        camtoworlds: torch.Tensor,
        Ks: torch.Tensor,
        width: int,
        height: int,
        frame_id: int = 0,
    ):
        """Rasterize the splats.
        Args:
            camtoworlds: [4,4]
            Ks: [3,3]
            width: int
            height: int
        Outputs:
            out: np.ndarray, rendered image
        """
        means = self.splats["means"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = self.splats["scales"] # [N, 3]
        opacities = self.splats["opacities"] # [N,]
        colors = self.splats["colors"]  # [N, 3]
        N = means.shape[0]

        means_copy = means.clone()

        # quantize
        means_tr, means_meta = quantize_attribute(means, bitdepth=16)
        quats_tr, quats_meta = quantize_attribute(quats)
        scales_tr, scales_meta = quantize_attribute(scales)
        opacities_tr, opacities_meta = quantize_attribute(opacities)
        colors_tr, colors_meta = quantize_attribute(colors)

        # pack into a single image
        gs_img = pack_3dgs_into_single_image(means_tr, quats_tr, scales_tr, colors_tr, opacities_tr)

        # # save packed frame
        # gs_img_dir = os.path.join(self.render_dir, "packed_frames")
        # save_packed_frame(gs_img, gs_img_dir, frame_id)

        # Load the decoded frames
        decoded_path = os.path.join(self.render_dir, "decoded_frames_lossless_ffv1", f"decoded_{frame_id+1:04d}.png")
        # decoded_path = os.path.join(self.render_dir, "decoded_frames_125M", f"decoded_{frame_id+1:04d}.png")
        gs_img  = load_decoded_frame(decoded_path, device=self.device)

        # unpack
        means_tr, quats_tr, scales_tr, colors_tr, opacities_tr = unpack_3dgs_from_single_image(gs_img, offset=N)
        
        # de-quantize
        means = dequantize_attribute(means_tr, means_meta, bitdepth=16)
        quats = dequantize_attribute(quats_tr, quats_meta)
        scales = dequantize_attribute(scales_tr, scales_meta)
        opacities = dequantize_attribute(opacities_tr, opacities_meta)
        colors = dequantize_attribute(colors_tr, colors_meta)

        # # Ablations:
        # means = means_copy  # ablation: use the original means

        viewmats = torch.linalg.inv(camtoworlds).unsqueeze(0)

        meta = rasterize_gaussian_images(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            camera_model=self.cfg.camera_model,
            sh_degree=None
        )
        meta['colors'] = meta['colors'].squeeze(0)
        render_colors, render_alphas, meta = rasterize_pixels(meta)
    
        # tile_list = group_gaussian_attributes_by_tiles(meta)
        # tile_idxs_t = tile_list[0]
        # colors_tile = meta["colors"][tile_idxs_t]
        # raise Exception(f"colors_tile: {colors_tile.shape}")

        render_colors = torch.clamp(render_colors, 0.0, 1.0)
        out = (render_colors[0].cpu().numpy() * 255).astype(np.uint8) 
        return out

    @torch.no_grad()
    def render_trajectory(self):
        """Render the GS acc to some trajectories.
        """

        camtoworlds_all = self.parser.camtoworlds  # e.g. shape (N,4,4)
        camtoworlds_all = camtoworlds_all[2:20]
        path_type = self.cfg.render_traj_path
        if path_type == "interp":
            camtoworlds_all = generate_interpolated_path(camtoworlds_all, 20)
        elif path_type == "ellipse":
            avg_z = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(camtoworlds_all, height=avg_z-0.2)
        elif path_type == "spiral":
            bounds = self.parser.bounds * self.scene_scale
            scale_r = self.parser.extconf.get("spiral_radius_scale", 1.0)
            camtoworlds_all = generate_spiral_path(camtoworlds_all, bounds=bounds, spiral_scale_r=scale_r)
        else:
            raise ValueError(f"Unknown path type: {path_type}")

        # Convert camtoworlds to (N,4,4)
        camtoworlds_all = np.concatenate([  
            camtoworlds_all,
            np.array([[[0,0,0,1]]]*len(camtoworlds_all))  # add the final row
        ], axis=1)
        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(self.device)
        camtoworlds_all = camtoworlds_all[30:60] # Crop the trajectory

        # Get intrinsics
        K_np = list(self.parser.Ks_dict.values())[0]  # pick the first
        K = torch.from_numpy(K_np).float().to(self.device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # Create video writer
        if self.save_video:
            video_path = os.path.join(self.video_dir, f"decompressed.mp4")
            writer = imageio.get_writer(video_path, fps=30)
            print(f"[SPTGSRenderer] Writing video to: {video_path}")
        else:
            writer = None

        # Main loop
        frames_count = len(camtoworlds_all)
        n_motion = 30
        frames_per_motion = frames_count // n_motion
        motion_indices = []
        for i in range(n_motion):
            motion_id = i / n_motion
            for j in range (frames_per_motion):
                motion_indices.append(motion_id)
            
        for i in trange(frames_count, desc="Rendering trajectory"):
            c2w = camtoworlds_all[i]  
            Ks = K[None]

            self.load_frame(self.splats, motion_indices[i])  # load the first frame
            out = self.rasterize_splats(c2w, Ks, width, height, frame_id=i)

            # save frame if requested
            if i == 0 and self.save_first_frame:
                # save only the first frame
                frame_path = os.path.join(self.render_dir, "frame_first.png")
                imageio.imwrite(frame_path, out)
            if self.save_frame:
                # save every frame
                frame_path = os.path.join(self.render_dir, f"frame_{i:04d}.png")
                imageio.imwrite(frame_path, out)

            # add to video
            if writer is not None:
                writer.append_data(out)

            # If we only want to save the first frame and skip the rest
            if self.save_first_frame and i == 0:
                break  # end the loop

        # close video
        if writer is not None:
            writer.close()
            print("[SPTGSRenderer] Video closed.")


def main():
    parser = argparse.ArgumentParser(description="Simplified rendering of GS.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the JSON file with rendering config.")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to the checkpoint directory or .ply file.")
    parser.add_argument("--save_video", action="store_true",
                        help="If set, save the entire trajectory as a .mp4 video.")
    parser.add_argument("--save_frame", action="store_true",
                        help="If set, save all frames as individual image files.")
    parser.add_argument("--save_first_frame", action="store_true",
                        help="If set, only save the very first frame.")
    args = parser.parse_args()

    cfg = load_config_json(args.config)
    renderer = DynamicGSRenderer(
        cfg=cfg,
        ckpt=args.ckpt,
        save_video=args.save_video,
        save_frame=args.save_frame,
        save_first_frame=args.save_first_frame,
    )
    # Render the trajectory
    renderer.render_trajectory()

if __name__ == "__main__":
    main()



