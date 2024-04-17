import os
import torch
import numpy as np
import random
import cv2

from torchvision import transforms
from models.maniqa import MANIQA
from torch.utils.data import DataLoader
from config import Config
from utils.inference_process import ToTensor, Normalize
from tqdm import tqdm
import argparse

from pathlib import Path
import pandas as pd


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Image(torch.utils.data.Dataset):
    def __init__(self, image_path, transform, num_crops=20):
        super(Image, self).__init__()
        self.img_name = image_path.split('/')[-1]
        self.img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img = np.array(self.img).astype('float32') / 255
        self.img = np.transpose(self.img, (2, 0, 1))

        self.transform = transform

        c, h, w = self.img.shape
        print(self.img.shape)
        new_h = 224
        new_w = 224

        self.img_patches = []
        for i in range(num_crops):
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            patch = self.img[:, top: top + new_h, left: left + new_w]
            self.img_patches.append(patch)
        
        self.img_patches = np.array(self.img_patches)

    def get_patch(self, idx):
        patch = self.img_patches[idx]
        sample = {'d_img_org': patch, 'score': 0, 'd_name': self.img_name}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_path', type=str)
    # directory
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--output', type=str)

    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    cpu_num = 3
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)

    # config file
    config = Config({
        # valid times
        "num_crops": 20,

        # model
        "patch_size": 8,
        "img_size": 224,
        "embed_dim": 768,
        "dim_mlp": 768,
        "num_heads": [4, 4],
        "window_size": 4,
        "depths": [2, 2],
        "num_outputs": 1,
        "num_tab": 2,
        "scale": 0.8,
    })
    
    # model defination
    net = MANIQA(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
        patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
        depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale,
        device=device)

    net.load_state_dict(torch.load(args.ckpt_path, map_location=device), strict=False)
    net = net.to(device)

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    results_df = pd.DataFrame(columns=['img_stem', 'score'])

    for p in Path(args.img_path).glob('*'):
        # data load
        Img = Image(image_path=str(p),
            transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
            num_crops=config.num_crops)

        avg_score = 0
        for i in tqdm(range(config.num_crops)):
            with torch.no_grad():
                net.eval()
                patch_sample = Img.get_patch(i)
                patch = patch_sample['d_img_org'].to(device)
                patch = patch.unsqueeze(0)
                score = net(patch)
                avg_score += score

        avg_score = avg_score / config.num_crops

        sample_dict = {
            'img_stem': p.stem,
            'score': avg_score.cpu().numpy()[0]
        }

        results_df.loc[len(results_df)] = pd.Series(sample_dict)
    
    results_df.to_csv(f'{args.output}/maniqa.csv', index=False)

    

    