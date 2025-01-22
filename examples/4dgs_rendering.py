import argparse
import math
import os
import time
from typing import Tuple
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import Literal, assert_never
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_interpolated_path,
    generate_ellipse_path_z,
    generate_spiral_path,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from typing_extensions import Literal
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from glob import glob
from plyfile import PlyData, PlyElement


@dataclass
class Config:
    exp_name: str = "4dgs"
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    ply_folder: str = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 1
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"
    

def load_ply(ply_file: str, feature_shape: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reads a PLY file and restores the elements: points, scales, rotations, opacity, and shs.
    Args:
        ply_file (str): Path to the PLY file.
        feature_shape (Tuple[int, int]): Tuple containing (feature_dc_channels, feature_rest_channels).

    Returns:
        points (torch.Tensor): [N, 3] Tensor containing Gaussian centers.
        scales (torch.Tensor): [N, 3] Tensor containing Gaussian scales.
        rotations (torch.Tensor): [N, 4] Tensor containing Gaussian rotations.
        opacities (torch.Tensor): [N] Tensor containing Gaussian opacities.
        shs (torch.Tensor): [N, K, 3] Tensor containing spherical harmonics coefficients.
    """
    # Load the PLY file
    ply_data = PlyData.read(ply_file)
    vertex_data = ply_data['vertex']

    # Extract basic attributes
    points = torch.tensor(np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=-1), dtype=torch.float32)
    scales = torch.tensor(np.stack([vertex_data['scale_0'], vertex_data['scale_1'], vertex_data['scale_2']], axis=-1), dtype=torch.float32)
    rotations = torch.tensor(np.stack([vertex_data[f'rot_{i}'] for i in range(4)], axis=-1), dtype=torch.float32)
    opacities = torch.tensor(vertex_data['opacity'], dtype=torch.float32)

    # Reconstruct spherical harmonics (shs)
    feature_dc_channels, feature_rest_channels = feature_shape
    sh_dc = np.stack([vertex_data[f'f_dc_{i}'] for i in range(feature_dc_channels * 3)], axis=-1).reshape(-1, feature_dc_channels, 3)
    sh_rest = np.stack([vertex_data[f'f_rest_{i}'] for i in range(feature_rest_channels * 3)], axis=-1).reshape(-1, feature_rest_channels, 3)
    shs = torch.cat([torch.tensor(sh_dc, dtype=torch.float32), torch.tensor(sh_rest, dtype=torch.float32)], dim=1)

    sh0 = shs[:, :1, :]
    shN = shs[:, 1:, :]
    params = [
        ("means", points),
        ("scales", scales),
        ("quats", rotations),
        ("opacities", opacities),
        ("sh0", sh0),
        ("shN", shN),
    ]
    splats = torch.nn.ParameterDict({n: v for n, v in params})
    return splats


class Runner:
    """Engine for training and testing."""
    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        self.stats_dir = f"{cfg.result_dir}/stats"
        self.render_dir = f"{cfg.result_dir}/renders"

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        # feature_dim = 32 if cfg.app_opt else None
        # self.splats = init_splats(
        #     self.parser,
        #     init_type=cfg.init_type,
        #     init_num_pts=cfg.init_num_pts,
        #     init_extent=cfg.init_extent,
        #     init_opacity=cfg.init_opa,
        #     init_scale=cfg.init_scale,
        #     scene_scale=self.scene_scale,
        #     sh_degree=cfg.sh_degree,
        #     feature_dim=feature_dim,
        #     device=self.device,
        #     world_rank=world_rank,
        #     world_size=world_size,
        # )
        self.splats = None
        self.ply_files = sorted(glob(os.path.join(cfg.ply_folder, "time_*.ply")))

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        
        means = self.splats["means"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        image_ids = kwargs.pop("image_ids", None)
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks, 
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    @torch.no_grad()
    def render_traj(self, frames: bool = False) -> None:
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device
        camtoworlds_all = self.parser.camtoworlds[3:10]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 20
            )  # [N, 3, 4]
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  
        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        video_path = f"{video_dir}/traj_{cfg.exp_name}.mp4"
        writer = imageio.get_writer(video_path, fps=30)
        if frames:
            render_dir = f"{cfg.result_dir}/renders"
            os.makedirs(render_dir, exist_ok=True)


        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            if i==0: 
                self.splats = load_ply(self.ply_files[0], [1, 15]).to(self.device)
                print("Model initialized. Number of GS:", len(self.splats["means"]))
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]

            
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)

            if frames and i:         
                colors = (colors.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                depths = depths.squeeze(0).cpu().numpy()
                np.save(f"{render_dir}/depth_{i:04d}.npy", depths[..., 0])
                depth_im = (depths * 255).astype(np.uint8)
                depth_im= np.tile(depth_im, (1, 1, 3))
                imageio.imwrite(f"{render_dir}/color_{i:04d}.png", colors)
                imageio.imwrite(f"{render_dir}/depth_{i:04d}.png", depth_im)
                # save camtoworlds
                np.save(f"{render_dir}/camtoworlds_{i:04d}.npy", camtoworlds.cpu().numpy())



def main(local_rank: int, world_rank, world_size: int, cfg: Config):

    runner = Runner(local_rank, world_rank, world_size, cfg)
    runner.render_traj(frames=False)

if __name__ == "__main__":
    root_dir = "/work/pi_rsitaram_umass_edu/tungi"
    ply_folder = os.path.join(root_dir, "4DGaussians/output/n3d/flame_steak/gaussian_pertimestamp")  
    configs = {
        "4dgs": (
            "4D Gaussian Splatting",
            Config(
                exp_name='4dgs',
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.render_traj_path = "interp"
    cfg.ply_folder = ply_folder
    cli(main, cfg, verbose=True)


# class Runner:
#     def __init__(self, local_rank: int, world_rank, world_size: int, cfg: Config) -> None:
#         self.cfg = cfg
#         self.world_rank = world_rank
#         self.local_rank = local_rank
#         self.world_size = world_size
#         self.device = f"cuda:{local_rank}"

#         # Setup output directories.
#         self.render_dir = f"{cfg.result_dir}/renders"
#         os.makedirs(self.render_dir, exist_ok=True)

#     @torch.no_grad()
#     def render_volumetric_video(self, ply_files: List[str], trajectory: Tensor) -> None:
#         print("Rendering volumetric video...")
#         video_path = f"{self.render_dir}/volumetric_video.mp4"
#         writer = imageio.get_writer(video_path, fps=30)

#         for i, camtoworld in enumerate(tqdm.tqdm(trajectory, desc="Rendering frames")):
#             # Load corresponding PLY file
#             frame_idx = min(i // 2, len(ply_files) - 1)
#             splats = load_ply(ply_files[frame_idx])

#             # Rasterize
#             renders, _, _ = rasterization(
#                 means=splats["means"],
#                 quats=splats["quats"],
#                 scales=torch.exp(splats["scales"]),
#                 opacities=torch.sigmoid(splats["opacities"]),
#                 colors=torch.cat([splats["sh0"], splats["shN"]], dim=1),
#                 viewmats=torch.linalg.inv(camtoworld[None, :, :]),
#                 Ks=torch.eye(3, device=self.device)[None, :, :],
#                 width=800,  # Adjust width and height as needed
#                 height=800,
#             )

#             # Prepare and write frame
#             canvas = torch.clamp(renders[..., :3], 0.0, 1.0).squeeze(0).cpu().numpy()
#             canvas = (canvas * 255).astype(np.uint8)
#             writer.append_data(canvas)

#         writer.close()
#         print(f"Video saved to {video_path}")


# def init_splats(
#     parser: Parser,
#     init_type: str = "sfm",
#     init_num_pts: int = 100_000,
#     init_extent: float = 3.0,
#     init_opacity: float = 0.1,
#     init_scale: float = 1.0,
#     scene_scale: float = 1.0,
#     sh_degree: int = 3,
#     feature_dim: Optional[int] = None,
#     device: str = "cuda",
#     world_rank: int = 0,
#     world_size: int = 1,
# ) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    
#     if init_type == "sfm":
#         points = torch.from_numpy(parser.points).float()
#         rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
#     elif init_type == "random":
#         points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
#         rgbs = torch.rand((init_num_pts, 3))
#     else:
#         raise ValueError("Please specify a correct init_type: sfm or random")

#     # Initialize the GS size to be the average dist of the 3 nearest neighbors
#     dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
#     dist_avg = torch.sqrt(dist2_avg)
#     scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

#     # Distribute the GSs to different ranks (also works for single rank)
#     points = points[world_rank::world_size]
#     rgbs = rgbs[world_rank::world_size]
#     scales = scales[world_rank::world_size]

#     N = points.shape[0]
#     quats = torch.rand((N, 4))  # [N, 4]
#     opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

#     params = [
#         ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
#         ("scales", torch.nn.Parameter(scales), 5e-3),
#         ("quats", torch.nn.Parameter(quats), 1e-3),
#         ("opacities", torch.nn.Parameter(opacities), 5e-2),
#     ]

#     if feature_dim is None:
#         # color is SH coefficients.
#         colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
#         colors[:, 0, :] = rgb_to_sh(rgbs)
#         params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
#         params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
#     else:
#         # features will be used for appearance and view-dependent shading
#         features = torch.rand(N, feature_dim)  # [N, feature_dim]
#         params.append(("features", torch.nn.Parameter(features), 2.5e-3))
#         colors = torch.logit(rgbs)  # [N, 3]
#         params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

#     splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
#     return splats