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
from gsplat.rendering import rasterization
from utils import knn, rgb_to_sh

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

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
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

class GSSimpleRenderer:
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

        # 1) Data parser for cameras
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.scene_scale = self.parser.scene_scale * cfg.global_scale
        print(f"[GSSimpleRenderer] Scene scale: {self.scene_scale:.4f}")

        # 2) Load the GS model from checkpoint
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
        file = f"{ckpt}/ckpts/ckpt_29999_rank0.pt"
        ckpts = [
            torch.load(file, map_location="cuda", weights_only=True)
        ]
        for k in self.splats.keys():
            self.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])

        # 3) Output directories
        self.result_dir = cfg.result_dir
        os.makedirs(self.result_dir, exist_ok=True)
        self.video_dir = os.path.join(self.result_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)
        self.render_dir = os.path.join(self.result_dir, "renders")
        os.makedirs(self.render_dir, exist_ok=True)

    @torch.no_grad()
    def rasterize_splats(
        self,
        camtoworlds: torch.Tensor,
        Ks: torch.Tensor,
        width: int,
        height: int,
    ) -> np.ndarray:
         
        means = self.splats["means"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        # rasterization returns (B,H,W,3 or 4)
        viewmats = torch.linalg.inv(camtoworlds).unsqueeze(0)
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,  # [B,4,4]
            Ks=Ks,              # [B,3,3]
            width=width,
            height=height,
            camera_model=self.cfg.camera_model,
            sh_degree=self.cfg.sh_degree
        )

        render_colors = torch.clamp(render_colors, 0.0, 1.0)
        out = (render_colors[0].cpu().numpy() * 255).astype(np.uint8) 
        return out

    @torch.no_grad()
    def render_trajectory(self):
        """
        Renders a path specified in the config: "render_traj_path" = "interp", "ellipse", "spiral", etc.
        """
        # get some subset from parser
        camtoworlds_all = self.parser.camtoworlds  # e.g. shape (N,4,4)
        # Example: take the first 10
        camtoworlds_all = camtoworlds_all[:10]

        path_type = self.cfg.render_traj_path
        if path_type == "interp":
            camtoworlds_all = generate_interpolated_path(camtoworlds_all, 20)
        elif path_type == "ellipse":
            avg_z = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(camtoworlds_all, height=avg_z)
        elif path_type == "spiral":
            bounds = self.parser.bounds * self.scene_scale
            scale_r = self.parser.extconf.get("spiral_radius_scale", 1.0)
            camtoworlds_all = generate_spiral_path(camtoworlds_all, bounds=bounds, spiral_scale_r=scale_r)
        else:
            raise ValueError(f"Unknown path type: {path_type}")

        # Convert to (N,4,4)
        camtoworlds_all = np.concatenate([
            camtoworlds_all,
            np.array([[[0,0,0,1]]]*len(camtoworlds_all))  # add the final row
        ], axis=1)
        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(self.device)

        # get intrinsics
        K_np = list(self.parser.Ks_dict.values())[0]  # pick the first
        K = torch.from_numpy(K_np).float().to(self.device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # if saving video => create writer
        if self.save_video:
            video_path = os.path.join(self.video_dir, f"traj_{path_type}.mp4")
            writer = imageio.get_writer(video_path, fps=30)
            print(f"[GSSimpleRenderer] Writing video to: {video_path}")
        else:
            writer = None

        # main loop
        frames_count = len(camtoworlds_all)
        for i in trange(frames_count, desc="Rendering trajectory"):
            c2w = camtoworlds_all[i]  
            Ks = K[None]

            out = self.rasterize_splats(c2w, Ks, width, height)

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
            print("[GSSimpleRenderer] Video closed.")


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
    renderer = GSSimpleRenderer(
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
