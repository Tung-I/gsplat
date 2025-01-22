import json
import math
import os
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import yaml
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from typing_extensions import Literal

# Minimal imports from your local code
from datasets.colmap import Dataset, Parser
from utils import set_random_seed, rgb_to_sh, knn
from fused_ssim import fused_ssim  # if you have a custom module for ssim


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    """
    Create a ParameterDict with 3D positions, scales (log-scale), quaternions,
    opacities, and spherical-harmonic colors from the sfm/pointcloud.
    """
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    else:
        # Just a random fallback
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))

    # Some approximate scale init
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4] not normalized
    opacities = torch.logit(torch.full((N,), init_opacity))  # raw logit

    # Spherical harmonics color
    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))
    # The first SH coefficient encodes the base color
    colors[:, 0, :] = rgb_to_sh(rgbs)

    # Convert to ParameterDict
    splats = torch.nn.ParameterDict({
        "means":     torch.nn.Parameter(points),
        "scales":    torch.nn.Parameter(scales),
        "quats":     torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0":       torch.nn.Parameter(colors[:, :1, :]),
        "shN":       torch.nn.Parameter(colors[:, 1:, :]),
    }).to(device)

    # Optimizers for each group
    # (You could unify them into one optimizer, but here we replicate your style)
    # Increase LR for means if you want faster geometry updates, etc.
    # The “lr scaling” logic is removed to keep it simple
    params_lr = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,
    }
    optimizers = {}
    for k in params_lr.keys():
        optimizers[k] = torch.optim.Adam([splats[k]], lr=params_lr[k], eps=1e-15)

    return splats, optimizers


@dataclass
class Config:
    # Data & general
    data_dir: str = "data/flame_steak/"
    data_factor: int = 1
    normalize_world_space: bool = True
    test_every: int = 8
    result_dir: str = "results/flame_steak_static"

    # Simple training
    max_steps: int = 30000
    batch_size: int = 1
    patch_size: Optional[int] = None

    # GS init
    init_type: str = "sfm"
    init_num_pts: int = 100000
    init_extent: float = 3.0
    init_opa: float = 0.1
    init_scale: float = 1.0
    sh_degree: int = 3

    # Loss weighting
    ssim_lambda: float = 0.2

    # Scene scale
    global_scale: float = 1.0

    # Evaluate steps & Save steps
    eval_steps: List[int] = field(default_factory=lambda: [7000, 30000])
    save_steps: List[int] = field(default_factory=lambda: [7000, 30000])

class Runner:
    def __init__(
        self, cfg: Config, local_rank: int = 0, world_rank: int = 0, world_size: int = 1
    ):
        self.cfg = cfg
        self.device = f"cuda:{local_rank}"
        self.world_rank = world_rank
        self.world_size = world_size

        os.makedirs(cfg.result_dir, exist_ok=True)
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(self.parser, split="train", patch_size=cfg.patch_size)
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Create a static GS model
        self.splats, self.optimizers = create_splats_with_optimizers(
            parser=self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            device=self.device,
        )
        print("Model init => number of GS:", len(self.splats["means"]))

        # Metrics
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

    def rasterize_splats(self, camtoworlds, Ks, width, height):
        """
        Minimal version of rasterization that uses your previous code snippet,
        or call your library's `rasterization(...)`.
        We'll just stub out a 'colors' result = random fill for demonstration.
        """
        B = camtoworlds.shape[0]
        # For demonstration, we do random fill: [B, H, W, 3].
        # In real usage, you'd replicate your GS rasterization from your pipeline.
        # Or if you have `gsplat.rendering.rasterization` => you can call it similarly.
        colors = torch.rand((B, height, width, 3), device=self.device)
        return colors

    def train(self):
        cfg = self.cfg
        max_steps = cfg.max_steps

        # Simple data loader
        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=4
        )
        trainloader_iter = iter(trainloader)

        for step in tqdm.trange(max_steps, desc="Training"):
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            # This data dict contains:
            #   "K": [B, 3, 3], "camtoworld": [B, 4, 4], "image": [B, H, W, 3], etc.
            camtoworlds = data["camtoworld"].to(self.device)
            Ks = data["K"].to(self.device)
            pixels = data["image"].to(self.device) / 255.0
            B, H, W, _ = pixels.shape

            # Rasterize
            colors = self.rasterize_splats(camtoworlds, Ks, W, H)
            # Compute L1 + SSIM
            l1loss = F.l1_loss(colors, pixels)
            ssim_val = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssim_val * cfg.ssim_lambda

            loss.backward()
            # Step each param-group
            for k, opt in self.optimizers.items():
                opt.step()
                opt.zero_grad(set_to_none=True)

            desc = f"loss={loss.item():.3f} (L1={l1loss.item():.3f}, SSIM={ssim_val.item():.3f})"
            if step % 500 == 0:
                tqdm.tqdm.write(desc)

            # Evaluate or save
            if step in cfg.eval_steps:
                self.eval(step)
            if step in cfg.save_steps:
                self.save_ckpt(step)

    @torch.no_grad()
    def eval(self, step: int):
        print("Running evaluation...")
        PSNRs = []
        SSIMs = []

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(self.device)
            Ks = data["K"].to(self.device)
            pixels = data["image"].to(self.device) / 255.0
            B, H, W, _ = pixels.shape

            colors = self.rasterize_splats(camtoworlds, Ks, W, H)
            colors = torch.clamp(colors, 0.0, 1.0)

            psnr_val = self.psnr(colors.permute(0,3,1,2), pixels.permute(0,3,1,2))
            ssim_val = self.ssim(colors.permute(0,3,1,2), pixels.permute(0,3,1,2))
            PSNRs.append(psnr_val)
            SSIMs.append(ssim_val)

            # Save images
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            imageio.imwrite(
                f"{self.render_dir}/val_step{step}_{i:03d}.png", (canvas * 255).astype(np.uint8)
            )

        # Summarize
        PSNR_val = torch.stack(PSNRs).mean().item()
        SSIM_val = torch.stack(SSIMs).mean().item()
        print(f"[Eval@step={step}] PSNR={PSNR_val:.3f}, SSIM={SSIM_val:.4f}")

    def save_ckpt(self, step: int):
        data = {"splats": self.splats.state_dict(), "step": step}
        torch.save(data, f"{self.ckpt_dir}/ckpt_{step}.pt")
        print(f"Saved checkpoint at step {step}.")

def main():
    cfg = Config()
    # Possibly load config from file or parse CLI
    set_random_seed(42)
    runner = Runner(cfg=cfg)
    runner.train()

if __name__ == "__main__":
    main()
