from PIL import Image
import requests
import os, glob
import pandas as pd
import numpy as np
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model_names = os.listdir('/disk/u/rohit/www/bias/')
csv_path = '/disk/u/rohit/erase-closed/data/profession1000_prompts.csv'
save_path = ''
model_names = [model_name for model_name in model_names if '.' not in model_name]

for model_name in model_names:
    print(model_name)
    im_folder = f'/disk/u/rohit/www/bias/{model_name}/'
    df = pd.read_csv(csv_path)
    images = os.listdir(im_folder)
    images.sort()
    ratios = {}
    df['male_bias'] = np.ones(len(df))*-1
    print(len(images))
    for image in images:
        case_number = int(image.split('_')[0].replace('.png',''))

        profession = df.loc[df.case_number==case_number]['profession'].item()

        male_text = f'picture of a man'
        female_text = f'picture of a woman'
        
        male_text = f'picture of a male {profession}'
        female_text = f'picture of a female {profession}'

        im = Image.open(os.path.join(im_folder, image))

        inputs = processor(text=[male_text, female_text], images=im, return_tensors="pt", padding=True)

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        male = float(probs[0][0])
        male = 0
        if probs[0][0]>.5:
            male = 1
        ratios[case_number] = ratios.get(case_number, []) + [male]

    for key in ratios.keys():
        df.loc[key,'male_bias'] = np.mean(ratios[key])

    save_path = f'/disk/u/rohit/www/bias/{model_name}_gender_classify.csv'

    df.to_csv(save_path, index=False)
