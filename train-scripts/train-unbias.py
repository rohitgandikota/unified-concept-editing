import numpy as np
import torch
import random
import pandas as pd
from PIL import Image
import pandas as pd 
import argparse

def view_images(images, num_rows=3, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    return pil_img


def diffusion_step(model, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (batch_size, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt,
    num_inference_steps = 50,
    guidance_scale = 7.5,
    generator = None,
    latent = None,
    low_resource = False,
):
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    
    model.scheduler.set_timesteps(num_inference_steps)
    for t in model.scheduler.timesteps:
        latents = diffusion_step(model, latents, context, t, guidance_scale, low_resource)
    
    image = latent2image(model.vae, latents)

#     image, _ = model.run_safety_checker(image=image, device=model.device, dtype=text_embeddings.dtype)
  
    return image

from PIL import Image
import requests
import os, glob
import pandas as pd
import numpy as np
from transformers import CLIPProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_ratios(ldm_stable, concept, classes, num_loops=3):
    seeds = [8393,47295,12321,0, 902]
    probs_full = []
    prompt = f'picture of {concept}'

    test_prompts = [f'picture of {class_}' for class_ in classes]
    with torch.no_grad():
        for seed in seeds:
#             if i == num_loops:
#                 break
            g = torch.Generator(device='cpu')
            g.manual_seed(seed)
            images = ldm_stable(prompt,num_images_per_prompt=10, num_inference_steps=20, generator = g).images

            inputs = clip_processor(text=test_prompts, images=images, return_tensors="pt", padding=True)

            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
            probs[probs<0.5] = 0
            probs[probs!=0] = 1
#             print(probs.mean(axis=0))
            probs_full.append(probs)
#     male = float(probs[0][0])
    return torch.cat(probs_full).mean(axis=0)

import torch
from diffusers import StableDiffusionPipeline
import numpy as np
import abc
import copy
import os

## get arguments for our script
with_to_k = False
with_augs = True
train_func = "train_closed_form"

### load model
LOW_RESOURCE = False
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77


def edit_model(ldm_stable, old_text_, new_text_, retain_text_, add=True, layers_to_edit=None, lamb=0.1, erase_scale = 0.1, preserve_scale = 0.1, with_to_k=True, adjust_to = 'Female'):
    ### collect all the cross attns modules
    sub_nets = ldm_stable.unet.named_children()
    ca_layers = []
    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__ :
                    for attn in block.attentions:
                        for  transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for  transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)

    ### get the value and key modules
    projection_matrices = [l.to_v for l in ca_layers]
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
    if with_to_k:
        projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
        og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_layers]

    ## reset the parameters
    num_ca_clip_layers = len(ca_layers)
    for idx_, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
            projection_matrices[num_ca_clip_layers + idx_] = l.to_k

    ### check the layers to edit (by default it is None; one can specify)
    layers_to_edit = ast.literal_eval(layers_to_edit) if type(layers_to_edit) == str else layers_to_edit
    lamb = ast.literal_eval(lamb) if type(lamb) == str else lamb

    ### Format the edits
    old_texts = []
    new_texts = []
    for old_text, new_text in zip(old_text_, new_text_):
        old_texts.append(old_text)
        n_t = []
        for t in new_text:

            if (old_text.lower() not in t.lower()) and add:
                n_t.append(t + ' ' +old_text)
            else:
                n_t.append(t)
        if len(n_t) == 1:
            n_t = n_t*2
        new_texts.append(n_t)
    if retain_text_ is None:
        ret_texts = ['']
        retain = False
    else:
        ret_texts = retain_text_
        retain = True

    desired_ratios = torch.ones(len(new_texts[0]))/len(new_texts[0])
    weight_step = 0.1
    weights = torch.zeros(len(new_texts[0]))
    aggregated = weights
    for i in range(20):
        ratios = get_ratios(ldm_stable, old_texts[0], new_texts[0])
        if i == 0 :
            init_ratios = ratios
        print(ratios)
        if (ratios - desired_ratios).abs().mean() < .05:
            print(weights)
            break
         #### restart LDM parameters
        num_ca_clip_layers = len(ca_layers)
        for idx_, l in enumerate(ca_layers):
            l.to_v = copy.deepcopy(og_matrices[idx_])
            projection_matrices[idx_] = l.to_v
            if with_to_k:
                l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
                projection_matrices[num_ca_clip_layers + idx_] = l.to_k

        weights_delta = weight_step * (desired_ratios - ratios)
        weights_delta[weights_delta<0] = -.05
        weights += weights_delta

        aggregated += weights_delta
        ### START EDIT


        for layer_num in range(len(projection_matrices)):
            if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
                continue

            #### prepare input k* and v*
            with torch.no_grad():
                #mat1 = \lambda W + \sum{v k^T}
                mat1 = lamb * projection_matrices[layer_num].weight

                #mat2 = \lambda I + \sum{k k^T}
                mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1], device = projection_matrices[layer_num].weight.device)

                for old_text, new_text in zip(old_texts, new_texts):
                    texts = [old_text]
                    texts = texts + new_text
                    text_input = ldm_stable.tokenizer(
                        texts,
                        padding="max_length",
                        max_length=ldm_stable.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                    old_emb = text_embeddings[0]

                    new_emb = text_embeddings[1:]

                    context = old_emb.detach()
                    values = []
                    with torch.no_grad():
                        for layer in projection_matrices:
                            embs = layer(new_emb[:]).detach()
                            o_embs = layer(old_emb).detach()

                            u1 = embs[0] #- embs[1]
                            u2 = embs[1]
                            u1 = u1 / u1.norm()
                            u2 = u2 / u2.norm()

                            o_emb_proj1 = (u1*o_embs).sum()
                            o_emb_proj2 = (u2*o_embs).sum()

                            max_proj = max(o_emb_proj1, o_emb_proj2)
                            target = o_embs + (weights[0]*o_embs.norm())*u1 + (weights[1]*o_embs.norm())*u2
    #                         max_proj = 0
#                             if adjust_to == 'Female':
#                                 target = o_embs + (-.05*o_embs.norm())*u1 + (.4*o_embs.norm())*u2 # - o_emb_proj2*u2 #- o_emb_proj1*u1 + max_proj*u1
    #                             target = o_embs + (max_proj-o_emb_proj1*1.1)*u1 + (max_proj*1.46-o_emb_proj2)*u2 # - o_emb_proj2*u2 #- o_emb_proj1*u1 + max_proj*u1
#                             else:
#                                 target = o_embs + (max_proj*2-o_emb_proj1)*u1 + (max_proj-o_emb_proj2*1.8)*u2 # - o_emb_proj2*u2 #- o_emb_proj1*u1 + max_proj*u1

    #                         print(o_emb_proj1.item(), (max_proj.item()-o_emb_proj1.item()*1.1)/o_embs.norm().item(), o_emb_proj2.item(), (max_proj.item()*1.46-o_emb_proj2.item())/o_embs.norm().item())

                            values.append(target.detach())
    #                         values.append(layer(target).detach())
    #                         values.append(layer(new_emb[:]).detach())
                    context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                    context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                    value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                    for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                    for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                    mat1 += erase_scale*for_mat1
                    mat2 += erase_scale*for_mat2

                for old_text, new_text in zip(ret_texts, ret_texts):
                    text_input = ldm_stable.tokenizer(
                        [old_text, new_text],
                        padding="max_length",
                        max_length=ldm_stable.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                    old_emb, new_emb = text_embeddings
                    context = old_emb.detach()
                    values = []
                    with torch.no_grad():
                        for layer in projection_matrices:
                            values.append(layer(new_emb[:]).detach())
                    context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                    context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                    value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                    for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                    for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                    mat1 += preserve_scale*for_mat1
                    mat2 += preserve_scale*for_mat2
                    #update projection matrix
                projection_matrices[layer_num].weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2))

    print(f'Current model status: Edited "{str(old_text_)}" into "{str(new_texts)}" and Retained "{str(retain_text_)}"')
    return ldm_stable, weights, init_ratios, ratios

def generate_for_text(ldm_stable, test_text, num_samples = 9, seed = 1231):
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    images = text2image_ldm_stable(ldm_stable, [test_text]*num_samples, latent=None, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=g, low_resource=LOW_RESOURCE)
    return view_images(images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainUSD',
                    description = 'Finetuning stable diffusion to debias the concepts')
    parser.add_argument('--concept', help='prompt corresponding to concept to erase', type=str, required=True)
    parser.add_argument('--attributes', help='Attributes to debias', type=str, default='Male, Female')
    parser.add_argument('--device', help='cuda devices to train on', type=str, required=False, default='0')
    args = parser.parse_args()

    device = f'cuda:{args.device}'
    concepts = args.concept
    concepts = args.concept.split(',')
    old_texts = [con.strip() for con in concepts]

    attributes = args.attributes
    attributes = attributes.split(',')
    attributes = [att.strip() for att in attributes]
    print_text = '-'
    for txt in  old_texts: 
        print_text += txt.lower()+'_'
    print_text  = print_text[:-1] + '-attributes-'
    for txt in attributes:
        print_text += txt.lower()+'_'
    print_text  = print_text[:-1]
    new_texts = [attributes for _ in concepts]
    
    df = pd.read_csv('data/profession1000_prompts.csv')

    retain_texts = list(df.profession.unique())
    
    old_texts_lower = [text.lower() for text in old_texts]
    retain_texts = [text for text in retain_texts if text.lower() not in old_texts_lower]
    print(old_texts, new_texts) 
    ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
    
    ldm_stable, weights, init_ratios, final_ratios = edit_model(ldm_stable= ldm_stable, old_text_= old_texts, new_text_=new_texts, add=True, retain_text_= retain_texts, lamb=0.5, erase_scale = 1, preserve_scale = .1)

    torch.save(ldm_stable.unet.state_dict(), f'models/unbiased{print_text}.pt')

    with open(f'data/unbiased{print_text}.txt', 'w') as fp:
        fp.write(str(weights)+'\n'+str(init_ratios)+'\n'+str(final_ratios))
