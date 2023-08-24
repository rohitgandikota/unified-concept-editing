from PIL import Image
import requests
import os, glob
import pandas as pd
import numpy as np
import re
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def sorted_nicely( l ):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


path = '/share/u/rohit/www/final_erase/coco/'
model_names = os.listdir(path)
model_names = [m for m in model_names if '.' not in m]
csv_path = '/share/u/rohit/erase-closed/prompts_dir/erased'
save_path = ''

model_names.sort()
if 'original' in model_names:
    model_names.remove('original')
model_names = [m for m in model_names if 'ssd' in m and ('10a' in m or '50a' in m)]
print(model_names)
#model_names = ['original'] + model_names
#model_names = [m for m in model_names if 'i2p' in m]
for model_name in model_names:
    print(model_name)
    csv_path = f'/share/u/rohit/erase-closed/data/coco_30k.csv'
    im_folder = os.path.join(path, model_name)
    df = pd.read_csv(csv_path)
    images = os.listdir(im_folder)
    images = sorted_nicely(images)
    ratios = {}
    df['clip'] = np.nan
    for image in images:
        try:
            case_number = int(image.split('_')[0].replace('.png',''))
            if case_number not in list(df['case_number']):
                continue
            caption = df.loc[df.case_number==case_number]['prompt'].item()
            im = Image.open(os.path.join(im_folder, image))
            inputs = processor(text=[caption], images=im, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            clip_score = outputs.logits_per_image[0][0].detach().cpu() # this is the image-text similarity score
            ratios[case_number] = ratios.get(case_number, []) + [clip_score]
        except:
            pass
    for key in ratios.keys():
        df.loc[key,'clip'] = np.mean(ratios[key])
    df = df.dropna(axis=0)
    print(f"Mean CLIP score: {df['clip'].mean()}")
    print('-------------------------------------------------')
    print('\n')
