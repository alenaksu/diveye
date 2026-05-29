from torch.utils import data
import torchvision.transforms as transforms
import torch
import os
from PIL import Image


class LSUIDataset(data.Dataset):
    """Paired underwater image dataset for LSUI and similar folder structures.

    Expected layout::

        data_dir/
            input/   (degraded underwater images)
            GT/      (ground-truth / reference images)

    Both folders must contain images with matching filenames.
    """

    def __init__(self, data_dir, training=True, crop_size=256):
        self.input_dir = os.path.join(data_dir, "input")
        self.gt_dir = os.path.join(data_dir, "GT")
        self.crop_size = crop_size
        self.training = training

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.names = sorted(
            f for f in os.listdir(self.input_dir)
            if os.path.splitext(f)[1].lower() in exts
        )
        assert len(self.names) > 0, f"No images found in {self.input_dir}"

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        raw = Image.open(os.path.join(self.input_dir, name)).convert("RGB")
        gt = Image.open(os.path.join(self.gt_dir, name)).convert("RGB")

        # Identical random crop for both images.
        # If the image is smaller than crop_size, pad it first so get_params
        # never raises "Required crop size larger than input image size".
        if self.crop_size is not None:
            pw = max(0, self.crop_size - raw.width)
            ph = max(0, self.crop_size - raw.height)
            if pw > 0 or ph > 0:
                raw = transforms.functional.pad(raw, (0, 0, pw, ph), padding_mode="reflect")
                gt  = transforms.functional.pad(gt,  (0, 0, pw, ph), padding_mode="reflect")
            i, j, h, w = transforms.RandomCrop.get_params(
                raw, output_size=(self.crop_size, self.crop_size)
            )
            raw = transforms.functional.crop(raw, i, j, h, w)
            gt = transforms.functional.crop(gt, i, j, h, w)

        # Identical random horizontal flip
        if self.training and torch.rand(1).item() > 0.5:
            raw = transforms.functional.hflip(raw)
            gt = transforms.functional.hflip(gt)

        raw = transforms.functional.to_tensor(raw)
        gt = transforms.functional.to_tensor(gt)
        return raw, gt, name


# ---------------------------------------------------------------------------
# Legacy FiveKDataset kept for compatibility with the original train.ipynb
# ---------------------------------------------------------------------------

class FiveKDataset(data.Dataset):
    """MIT-Adobe FiveK paired dataset."""

    def __init__(self, list_file, raw_dir, expert_dir, training, size=None, filenames=False):
        join = os.path.join
        self.file_list = []
        with open(list_file) as f:
            for line in f:
                name = line.strip()
                if name:
                    p = (join(raw_dir, name), join(expert_dir, name), name)
                    self.file_list.append(p)
        self.filenames = filenames
        transformation = []
        if size is not None:
            transformation.append(transforms.Resize((size, size)))
        if training:
            transformation.append(transforms.RandomHorizontalFlip(0.5))
        transformation.append(transforms.ToTensor())
        self.transform = transforms.Compose(transformation)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        raw = Image.open(self.file_list[index][0]).convert("RGB")
        expert = Image.open(self.file_list[index][1]).convert("RGB")
        raw = self.transform(raw)
        expert = self.transform(expert)
        if self.filenames:
            return raw, expert, self.file_list[index][2]
        return raw, expert
