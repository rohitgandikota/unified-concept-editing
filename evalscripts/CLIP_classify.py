from PIL import Image
import requests
import os, glob
import pandas as pd
import numpy as np
import re
from transformers import CLIPProcessor, CLIPModel
import argparse

def sorted_nicely( l ):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def CLIP_classification(im_path, attributes, prompts_path, save_path, from_case, till_case):

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model_name = os.path.basename(os.path.abspath(im_path))
    print(model_name, attributes, save_path)
    df = pd.read_csv(prompts_path)
    images = os.listdir(im_path)
    images = [im for im in images if '.png' in im]
    images = sorted_nicely(images)
    ratios = {}
    columns = [f"{att.replace(' ','_')}_bias" for att in attributes]
    for col in columns:
        df[col] = np.nan
    for image in images:
        try:
            case_number = int(image.split('_')[0].replace('.png',''))
            if case_number < from_case or case_number > till_case:
                continue

            im = Image.open(os.path.join(im_path, image))

            inputs = processor(text=attributes, images=im, return_tensors="pt", padding=True)

            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
            tmax = probs.max(1, keepdim=True)[0]
            mask = list(probs.ge(tmax)[0].float().numpy())
            ratios[case_number] = ratios.get(case_number, []) + [mask]
        except Exception:
            ratios[case_number] = ratios.get(case_number, []) + [[0]*len(attributes)]

    for key in ratios.keys():
        print(np.array(ratios[key]))
        for idx, col in enumerate(columns):
            df.loc[key,col] = np.mean(np.array(ratios[key])[:,idx])

    save_path_ = f'{save_path}/{model_name}_gender_classify.csv'

    df.to_csv(save_path_, index=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'CLIP classification',
                    description = 'Takes the path to images and gives CLIP classification scores')
    parser.add_argument('--im_path', help='path to images', type=str, required=True)
    parser.add_argument('--attributes', help='comma separated attributes to classify against', type=str, required=False, default='a man,a woman')
    parser.add_argument('--prompts_path', help='path to csv prompts', type=str, required=True)
    parser.add_argument('--save_path', help='path to save results', type=str, required=False, default=None)
    parser.add_argument('--from_case', help='case number start', type=int, required=False, default=0)
    parser.add_argument('--till_case', help='case number end', type=int, required=False, default=1000000000)
    args = parser.parse_args()
    
    im_path = args.im_path
    attributes = [attrib.strip() for attrib in args.attributes.split(',')]
    prompts_path = args.prompts_path
    save_path = args.save_path
    from_case = args.from_case
    till_case = args.till_case
    
    if save_path is None:
        save_path = im_path.replace(os.path.basename(os.path.abspath(im_path)),'')
    
    CLIP_classification(im_path=im_path, attributes=attributes, prompts_path=prompts_path, save_path=save_path, from_case=from_case, till_case=till_case)