import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
import json

from CSD.model import CSD_CLIP
from CSD.utils import has_batchnorms, convert_state_dict
from CSD.loss_utils import transforms_branch0

torch.backends.cudnn.benchmark = True


class ImgDataset(Dataset):
    def __init__(self, paths, indices, transform):
        self.paths = paths
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), int(self.indices[i])


class CSD_Eval:
    def __init__(
        self,
        ckpt_path: str,
        refer_path: str,
        device: str = "cuda:0",
        compile_model: bool = False,
    ):
        self.device = torch.device(device)

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

        self.ref_embeds = torch.load(refer_path).to(self.device)
        self.ref_embeds = F.normalize(self.ref_embeds, dim=-1)

    def cal_score(
        self,
        img_groups,
        block_size: int = 500,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        N = len(img_groups)
        assert N == self.ref_embeds.size(0)

        flat_paths = []
        flat_indices = []
        for i, paths in enumerate(img_groups):
            if len(paths) == 0:
                raise ValueError(f"Index {i} has no images")
            for p in paths:
                flat_paths.append(p)
                flat_indices.append(i)

        ds = ImgDataset(flat_paths, flat_indices, self.preprocess)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
        )

        feats_list = []
        idx_list = []
        with torch.no_grad():
            for imgs, idx in tqdm(loader, leave=False):
                imgs = imgs.to(self.device, non_blocking=True)
                idx = idx.to(self.device, non_blocking=True)
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16):
                    out = self.model(imgs)[0]
                feats = F.normalize(out, dim=-1)
                feats_list.append(feats)
                idx_list.append(idx)

        feats_all = torch.cat(feats_list, dim=0).to(self.ref_embeds.dtype)
        idx_all = torch.cat(idx_list, dim=0).long()

        scores = torch.full(
            (N,), -float("inf"), device=self.device, dtype=feats_all.dtype
        )

        for start in range(0, N, block_size):
            end = min(start + block_size, N)
            size = end - start

            ref_blk = self.ref_embeds[start:end]
            mask = (idx_all >= start) & (idx_all < end)
            if not torch.any(mask):
                continue

            feats_blk = feats_all[mask]
            idx_blk = (idx_all[mask] - start).long()

            sims = feats_blk @ ref_blk.t()
            row = torch.arange(sims.size(0), device=self.device)
            self_col = sims[row, idx_blk]
            sum_all = sims.sum(dim=1)
            mean_sims = (sum_all - self_col) / (size - 1)

            block_max = torch.full(
                (size,), -float("inf"), device=self.device, dtype=mean_sims.dtype
            )
            for j in range(size):
                m = idx_blk == j
                if torch.any(m):
                    block_max[j] = mean_sims[m].max()

            scores[start:end] = block_max

        if torch.isinf(scores).any():
            raise ValueError("Some indices had no valid candidates")

        return scores.mean().cpu().item()


def group_images_by_index(img_paths, n_expected):
    groups = [[] for _ in range(n_expected)]
    for p in img_paths:
        parent = p.parent.name
        nums = re.findall(r"\d+", parent)
        if not nums:
            continue
        idx = int(nums[-1])
        if 0 <= idx < n_expected:
            groups[idx].append(p)
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--refer_path", type=str, required=True)
    parser.add_argument("--block_size", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    evaluator = CSD_Eval(
        ckpt_path=args.ckpt_path,
        refer_path=args.refer_path,
        device=args.device,
    )

    n_expected = evaluator.ref_embeds.size(0)

    for md in sorted(Path(args.dir).iterdir()):
        if not md.is_dir():
            continue

        img_paths = sorted(md.rglob("*.png"))
        img_groups = group_images_by_index(img_paths, n_expected)

        missing = [i for i, g in enumerate(img_groups) if len(g) == 0]
        if missing:
            raise ValueError(f"{md.name} missing indices: {missing[:10]}")

        score = evaluator.cal_score(
            img_groups,
            block_size=args.block_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        save_path = md.parent / f"{md.name}_scores.json"
        with open(save_path, "w") as f:
            json.dump({"scores": score}, f)

        print(f"[{md.name}] computed scores: {score}")


if __name__ == "__main__":
    main()
