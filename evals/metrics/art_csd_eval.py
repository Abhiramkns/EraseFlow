import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
import json

from CSD.model import CSD_CLIP
from CSD.utils import has_batchnorms, convert_state_dict
from CSD.loss_utils import transforms_branch0

# Enable cuDNN auto-tuner for potentially faster convolutions
torch.backends.cudnn.benchmark = True


class ImgDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img)


class CSD_Eval:
    def __init__(
        self,
        ckpt_path: str,
        refer_path: str,
        device: str = "cuda:0",
        compile_model: bool = False,
    ):
        self.device = torch.device(device)

        # Load model
        model = CSD_CLIP("vit_large", content_proj_head="default")
        if has_batchnorms(model):
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = convert_state_dict(ckpt["model_state_dict"])
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        if compile_model and hasattr(torch, "compile"):
            model = torch.compile(model)

        self.model = model.to(self.device)
        self.preprocess = transforms_branch0

        # Load and normalize reference embeddings (float32)
        self.ref_embeds = torch.load(refer_path).to(self.device)
        self.ref_embeds = F.normalize(self.ref_embeds, dim=-1)

    def cal_score(
        self,
        img_paths,
        block_size: int = 500,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        # 1) DataLoader for async loading + preprocessing
        ds = ImgDataset(img_paths, self.preprocess)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
        )

        # 2) Extract and normalize all features
        feats_list = []
        with torch.no_grad():
            for batch in tqdm(loader, leave=False):
                batch = batch.to(self.device)
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16):
                    out = self.model(batch)[0]
                feats = F.normalize(out, dim=-1)
                feats_list.append(feats)

        feats_all = torch.cat(feats_list, dim=0)                    # [N, D]
        feats_all = feats_all.to(self.ref_embeds.dtype)            # float32
        N, D = feats_all.shape
        print(f"Featues of shape: {N}")
        assert N == self.ref_embeds.size(0), \
            f"Number of images ({N}) must match refer_embeds length ({self.ref_embeds.size(0)})"

        # 3) Block-wise similarity: per-feature mean excluding diagonal
        scores = torch.empty(N, device=self.device, dtype=feats_all.dtype)
        for start in range(0, N, block_size):
            end = min(start + block_size, N)
            size = end - start

            feats_blk = feats_all[start:end]                       # [size, D]
            ref_blk   = self.ref_embeds[start:end]                 # [size, D]
            sims      = feats_blk @ ref_blk.t()                    # [size, size]

            diag      = sims.diagonal()                            # [size]
            sum_all   = sims.sum(dim=1)                            # [size]
            mean_sims = (sum_all - diag) / (size - 1)               # [size]

            scores[start:end] = mean_sims

        return scores.mean().cpu().item()


def main():
    parser = argparse.ArgumentParser(
        description="Block-wise similarity evaluation script"
    )
    parser.add_argument(
        "--dir", type=str, required=True,
        help="Root directory containing one subfolder per model"
    )
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument(
        "--refer_path", type=str,
        required=True
    )
    parser.add_argument(
        "--block_size", type=int, default=500,
        help="Size of each block for comparison"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    evaluator = CSD_Eval(
        ckpt_path=args.ckpt_path,
        refer_path=args.refer_path,
        device=args.device,
    )

    for md in sorted(Path(args.dir).iterdir()):
        if not md.is_dir():
            continue

        img_paths = sorted(md.rglob("*.png"))
        scores = evaluator.cal_score(
            img_paths,
            block_size=args.block_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        save_path = md.parent / f"{md.name}_scores.json"
        with open(save_path, "w") as f:
            json.dump({"scores": scores}, f)

        print(f"[{md.name}] computed scores: {scores}")


if __name__ == "__main__":
    main()