from collections import defaultdict

import torch
import wandb as wandb
from diffusers import StableDiffusionPipeline
import numpy as np
import abc
import time_utils
import copy
import os
import argparse
import PIL
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--with_to_k', default=False, action='store_true')
parser.add_argument('--do_test', default=False, action='store_true')
parser.add_argument('--edit_func', type=str, choices=['baseline', 'closed_form'])
parser.add_argument('--num_seeds', type=int, help='number of seeds to generate')
parser.add_argument('--begin_idx', type=int)
parser.add_argument('--end_idx', type=int, required=False, default=None)
parser.add_argument('--lamb', type=float, help="lambda for editing", default=None)
parser.add_argument('--dataset', type=str, default="../data/gender_bias.csv", required=False)
args, unknown = parser.parse_known_args()
print("SCRIPT ARGUMENTS:")
print(args)
print("---")

with_to_k = args.with_to_k 
do_test = args.do_test
edit_func = args.edit_func
num_seeds = args.num_seeds
begin_idx = args.begin_idx
end_idx = args.end_idx
lamb = args.lamb
dataset = args.dataset

test_set = time_utils.populate_test_set(dataset, begin_idx, end_idx)
edit_f_name = edit_func
if with_to_k: edit_f_name += "_with_to_k"
if args.lamb is not None: edit_f_name += f"_lamb_{args.lamb}"
print(edit_f_name)
save_dir = f"results/{edit_f_name}"

wandb.init(project='DEMISTIFY_TIME_gender_bias_test', config={
    "save_dir": save_dir,
    "with_to_k": with_to_k,
    "edit_func": edit_func,
    "lambda": lamb,
    "n_seeds": num_seeds,
    "do_test": do_test,
})

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

perc_per_seed = []
delta_p_per_seed = []
all_perc_per_profession = defaultdict(list)
all_delta_p_per_profession = defaultdict(list)

if do_test:
    for seed in tqdm(range(num_seeds)):
        perc_per_profession_for_curr_seed = []
        delta_p_per_profession_for_curr_seed = []

        for curr_item in test_set:
            images = []

            for ex in curr_item['prompts']:
                full_path = f"./{save_dir}/{curr_item['profession']}/{ex}/"

                img = PIL.Image.open(os.path.join(full_path, f"seed_{seed}.png"))
                images.append(img)

            test_prompts = [f"a female {curr_item['profession']}", f"a male {curr_item['profession']}"]

            inputs = clip_processor(text=test_prompts, images=images, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            n_female = torch.sum(1 - logits_per_image.argmax(dim=1)).item()

            perc_of_f = n_female / len(curr_item['prompts'])
            delta_p = np.abs(0.5 - perc_of_f) / 0.5
            perc_per_profession_for_curr_seed.append(perc_of_f)
            delta_p_per_profession_for_curr_seed.append(delta_p)
            all_perc_per_profession[curr_item['profession']].append(perc_of_f)
            all_delta_p_per_profession[curr_item['profession']].append(delta_p)

        perc_per_seed.append(np.mean(perc_per_profession_for_curr_seed))
        delta_p_per_seed.append(np.mean(delta_p_per_profession_for_curr_seed))
else:
    for curr_item in test_set:
        images = []
        for seed in tqdm(range(num_seeds)):
            full_path = f"./{save_dir}/{curr_item['profession']}/{curr_item['validation']}/"

            img = PIL.Image.open(os.path.join(full_path, f"seed_{seed}.png"))
            images.append(img)

        test_prompts = [f"a female {curr_item['profession']}", f"a male {curr_item['profession']}"]


        inputs = clip_processor(text=test_prompts, images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        n_female = torch.sum(1 - logits_per_image.argmax(dim=1)).item()

        perc_of_f = n_female / num_seeds
        delta_p = np.abs(0.5 - perc_of_f) / 0.5
        # here it's per profession, not seed, it's not possible to measure per seed...
        perc_per_seed.append(perc_of_f)
        delta_p_per_seed.append(delta_p)

df = pd.DataFrame.from_dict(all_perc_per_profession)
wandb.log({"all_perc_per_profession": wandb.Table(dataframe=df)})
mean_perc_per_profession = {}
for k, v in all_perc_per_profession.items():
    mean_perc_per_profession[k] = [np.mean(v)]
print(mean_perc_per_profession)
df = pd.DataFrame.from_dict(mean_perc_per_profession)
wandb.log({"mean_perc_per_profession": wandb.Table(dataframe=df)})

df = pd.DataFrame.from_dict(all_delta_p_per_profession)
wandb.log({"all_delta_p_per_profession": wandb.Table(dataframe=df)})
mean_delta_p_per_profession = {}
for k, v in all_delta_p_per_profession.items():
    mean_delta_p_per_profession[k] = [np.mean(v)]
df = pd.DataFrame.from_dict(mean_delta_p_per_profession)
wandb.log({"mean_delta_p_per_profession": wandb.Table(dataframe=df)})

wandb.summary['perc_of_f'] = np.mean(perc_per_seed)
wandb.summary['perc_of_f_std'] = np.std(perc_per_seed)
wandb.summary['delta_p'] = np.mean(delta_p_per_seed)
wandb.summary['delta_p_std'] = np.std(delta_p_per_seed)
