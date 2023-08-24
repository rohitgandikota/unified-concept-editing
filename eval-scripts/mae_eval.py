from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import copy
import os
import pandas as pd
import argparse
from dreamsim import dreamsim

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'LPIPS',
                    description = 'Takes the path to two images and gives LPIPS')
    parser.add_argument('--original_path', help='path to original image', type=str, required=True)
    parser.add_argument('--edited_path', help='path to edited image', type=str, required=True)
    parser.add_argument('--csv_path', help='path to csv prompts', type=str, required=True)
    parser.add_argument('--save_path', help='path to save results', type=str, required=False, default=None)
    parser.add_argument('--device', help='path to save results', type=int, required=False, default=0)
    parser.add_argument(
        "--image",
        action="store_true",
        help="Whether it is a single image path",
    )
    
    args = parser.parse_args()
    if True:
        file_names = os.listdir(args.original_path)
        file_names = [name for name in file_names if '.png' in name]
        df_prompts = pd.read_csv(args.csv_path)
        
        df_prompts['dream_loss'] = df_prompts['case_number'] *0
        for index, row in df_prompts.iterrows():
            case_number = row.case_number
            files = [file for file in file_names if file.startswith(f'{case_number}_')]
            lpips_scores = []
            for file in files:
                print(file)
                try:
                    img1 = Image.open(os.path.join(args.original_path,file))
                    img2 = Image.open(os.path.join(args.edited_path,file))
                    
                    l = np.sum(np.absolute((img1.astype("float") - img2.astype("float")))
                    print(f'Dreamsim score: {l}')
                    lpips_scores.append(l)
                except Exception as e:
                    print(f'No File : {e}')
                    pass
            df_prompts.loc[index,'dream_loss'] = np.mean(lpips_scores)
        if args.save_path is not None:
            if len(os.path.basename(args.edited_path).strip()) == 0:
                basename = args.edited_path.split('/')[-2]
            else:
                basename = args.edited_path.split('/')[-1]
            df_prompts.to_csv(os.path.join(args.save_path, f'{basename}_maeloss.csv'))
# python eval-scripts/lpips_eval.py --original_path '/share/u/rohit/www/closed_form/niche_short/original/' --csv_path '/share/u/rohit/erase-closed/data/short_niche_art_prompts.csv' --save_path '/share/u/rohit/www/closed_form/niche_short/' --edited_path '/share/u/rohit/www/closed_form/niche_short/erasing-ThomasKinkade-with-preservation/'
