"""
Embed a folder of .svs whole-slide images with Virchow2.

Tiles at 20x (0.5 mpp), 224 px, and drops tiles without enough tissue, following the PRISM2
paper (arXiv:2506.13063, Methods 4.1). The paper uses a tissue segmentation model. Here a
simple HSV threshold stands in for it.

--wsi-dir is searched recursively, so a flat folder and the GDC download layout (one UUID
subfolder per slide) both work. All paths come from CLI args.

Needs openslide-python and opencv-python, plus the OpenSlide system library.

Usage:
    python embed_wsi.py --wsi-dir /path/to/wsi_folder --output-dir /path/to/save/embeddings
    python embed_wsi.py --wsi-dir /path/to/wsi_folder --output-dir /path/to/save/embeddings --label my_label
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import openslide
import torch
import timm
from PIL import Image as PILImage
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed a folder of WSIs (.svs) with Virchow2")
    parser.add_argument("--wsi-dir", required=True, type=Path,
                         help="folder containing .svs files (searched recursively)")
    parser.add_argument("--output-dir", required=True, type=Path,
                         help="folder to save the resulting *_tiles.pt embeddings into")
    parser.add_argument("--label", default="Unknown",
                         help="label stored with every slide (default: Unknown — raw WSI "
                              "downloads rarely carry a label up front)")
    # defaults follow the PRISM2 paper
    parser.add_argument("--target-mpp", type=float, default=0.5,
                         help="target microns-per-pixel; 0.5 = 20x (default: 0.5, matches PRISM2 paper)")
    parser.add_argument("--tile-size", type=int, default=224, help="tile size in pixels (default: 224)")
    parser.add_argument("--min-tissue", type=float, default=0.65,
                         help="minimum tissue fraction to keep a tile "
                              "(default: 0.65, matches PRISM2 paper Methods 4.1)")
    parser.add_argument("--batch-size", type=int, default=64,
                         help="tiles per GPU forward pass (default: 64)")
    return parser.parse_args()


def load_virchow2(device: torch.device) -> tuple[torch.nn.Module, object]:
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    ).eval().to(device)
    transform = timm.data.create_transform(
        **timm.data.resolve_data_config(model.pretrained_cfg, model=model)
    )
    return model, transform


def get_best_level(
    slide: openslide.OpenSlide, target_mpp: float, tile_size: int
) -> tuple[int, float, int]:
    """Return (level_idx, actual_mpp, read_size) for the coarsest level whose mpp is still <= target_mpp."""
    mpp0 = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))
    best_level = 0
    for i in range(slide.level_count):
        level_mpp = mpp0 * slide.level_downsamples[i]
        if level_mpp <= target_mpp * 1.05:   # 5% tolerance for scanner rounding
            best_level = i
        else:
            break
    actual_mpp = mpp0 * slide.level_downsamples[best_level]
    read_size  = max(tile_size, round(tile_size * target_mpp / actual_mpp))
    return best_level, actual_mpp, read_size


def extract_tiles_svs(
    slide: openslide.OpenSlide, level: int, tile_size: int, read_size: int, min_tissue: float,
) -> list[PILImage.Image]:
    """Return tiles with enough tissue, using an HSV mask on a low-res level (stand-in for the paper's FCN segmenter)."""
    w, h     = slide.level_dimensions[level]
    ds_level = slide.level_downsamples[level]

    target_ds  = ds_level * 16
    lr_level   = min(range(slide.level_count), key=lambda i: abs(slide.level_downsamples[i] - target_ds))
    lr_ds      = slide.level_downsamples[lr_level]
    lr_w, lr_h = slide.level_dimensions[lr_level]

    lr_img      = slide.read_region((0, 0), lr_level, (lr_w, lr_h)).convert("RGB")
    hsv         = cv2.cvtColor(np.array(lr_img), cv2.COLOR_RGB2HSV)
    H, S, V     = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    tissue_mask = (H >= 90) & (H <= 180) & (S >= 8) & (V >= 103)

    relative_ds  = lr_ds / ds_level
    tile_size_lr = max(1, round(read_size / relative_ds))

    tiles: list[PILImage.Image] = []
    n_x, n_y = w // read_size, h // read_size
    for yi in range(n_y):
        for xi in range(n_x):
            mx0, my0 = xi * tile_size_lr, yi * tile_size_lr
            patch    = tissue_mask[my0:my0 + tile_size_lr, mx0:mx0 + tile_size_lr]
            if patch.size == 0 or patch.mean() < min_tissue:
                continue
            x0     = int(xi * read_size * ds_level)
            y0     = int(yi * read_size * ds_level)
            region = slide.read_region((x0, y0), level, (read_size, read_size)).convert("RGB")
            if read_size != tile_size:
                region = region.resize((tile_size, tile_size), PILImage.BILINEAR)
            tiles.append(region)
    return tiles


def embed_tiles_batch(
    imgs: list[PILImage.Image],
    model: torch.nn.Module,
    transform,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """PIL tiles -> (N, 1280) CLS embeddings."""
    all_embs = []
    for i in range(0, len(imgs), batch_size):
        batch = torch.stack([transform(img) for img in imgs[i:i + batch_size]]).to(device)
        with torch.inference_mode():
            all_embs.append(model.forward_features(batch)[:, 0].cpu())   # (B, 1280)
    return torch.cat(all_embs, dim=0)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"WSI_DIR    : {args.wsi_dir}")
    print(f"OUTPUT_DIR : {args.output_dir}")
    print(f"LABEL      : {args.label}")

    model, transform = load_virchow2(device)
    print(f"Virchow2 params: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    svs_files = sorted(args.wsi_dir.rglob("*.svs"))
    print(f"Found {len(svs_files)} SVS files in {args.wsi_dir}")

    skipped_empty = []
    for svs_path in tqdm(svs_files, desc="Slides"):
        slide_id  = svs_path.stem
        save_path = args.output_dir / f"{slide_id}_tiles.pt"
        if save_path.exists():
            continue

        slide = openslide.OpenSlide(str(svs_path))
        level, _, read_size = get_best_level(slide, args.target_mpp, args.tile_size)
        tiles = extract_tiles_svs(slide, level, args.tile_size, read_size, args.min_tissue)
        slide.close()

        if not tiles:
            print(f"  Skipped (no tissue tiles): {slide_id}")
            skipped_empty.append(slide_id)
            continue

        tile_embs = embed_tiles_batch(tiles, model, transform, device, args.batch_size)
        torch.save({"features": tile_embs, "label": args.label}, save_path)

    n_done = len(list(args.output_dir.glob("*_tiles.pt")))
    print(f"Done — {n_done} files in {args.output_dir}")
    if skipped_empty:
        print(f"Skipped (no tissue): {skipped_empty}")


if __name__ == "__main__":
    main()
