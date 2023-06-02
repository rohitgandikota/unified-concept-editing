import numpy as np
import torch
import random
import pandas as pd
from PIL import Image
import torch
import ast
from diffusers import StableDiffusionPipeline
import abc
import copy
import os
def view_images(images, num_rows=1, offset_ratio=0.02):
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
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
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

def edit_model(old_text_, new_text_, retain_text_,layers_to_edit=None, lamb=0.1, preserve_scale=0.1):
    #### restart LDM parameters
    num_ca_clip_layers = len(ca_clip_layers)
    for idx_, l in enumerate(ca_clip_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
            projection_matrices[num_ca_clip_layers + idx_] = l.to_k
    layers_to_edit = ast.literal_eval(layers_to_edit) if type(layers_to_edit) == str else layers_to_edit
    lamb = ast.literal_eval(lamb) if type(lamb) == str else lamb

    for layer_num in range(len(projection_matrices)):
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue
#     try:
    #### set up sentences
        old_texts = old_text_
        new_texts = []
        for old_text in old_texts:
            new_texts.append(new_text_)

        if retain_text_ is None:
            retain_text__ = ['']
        else:
            retain_text__ = retain_text_
        ret_texts = retain_text__
        ret_new_texts = []
        for ret_text in ret_texts:
            ret_new_texts.append(ret_text)
        retain = False
        if retain_text_ is not None:
            retain = True
        #### prepare input k* and v*
        with torch.no_grad():
            #mat1 = \lambda W + \sum{v k^T}
            mat1 = lamb * projection_matrices[layer_num].weight

            #mat2 = \lambda I + \sum{k k^T}
            mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1], device = projection_matrices[layer_num].weight.device)

            for old_text, new_text in zip(old_texts, new_texts):
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
                mat1 += for_mat1
                mat2 += for_mat2

            for old_text, new_text in zip(ret_texts, ret_new_texts):
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
#     #### define training function
#     train = TRAIN_FUNC_DICT[train_func]
    
    
#     #### train the model
#     train(ldm_stable, projection_matrices, og_matrices, contexts, valuess, old_texts,
#           new_texts, ret_contexts, ret_valuess, ret_texts,
#           ret_new_texts, lamb=lamb, retain = retain)

    return f'Current model status: Edited "{str(old_text_)}" into "{str(new_text_)}" and Retained "{str(retain_text_)}"'

def generate_for_text(test_text, seed = 1231):
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    images = text2image_ldm_stable(ldm_stable, [test_text], latent=None, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=g, low_resource=LOW_RESOURCE)
    return view_images(images)

with_to_k = True
with_augs = True
train_func = "train_closed_form"

### load model
LOW_RESOURCE = True
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
tokenizer = ldm_stable.tokenizer

### get layers
ca_layers = []
def append_ca(net_):
    if net_.__class__.__name__ == 'CrossAttention':
        ca_layers.append(net_)
    elif hasattr(net_, 'children'):
        for net__ in net_.children():
            append_ca(net__)

sub_nets = ldm_stable.unet.named_children()
for net in sub_nets:
        if "down" in net[0]:
            append_ca(net[1])
        elif "up" in net[0]:
            append_ca(net[1])
        elif "mid" in net[0]:
            append_ca(net[1])

### get projection matrices
ca_clip_layers = [l for l in ca_layers if l.to_v.in_features == 768]
projection_matrices = [l.to_v for l in ca_clip_layers]
og_matrices = [copy.deepcopy(l.to_v) for l in ca_clip_layers]
if with_to_k:
    projection_matrices = projection_matrices + [l.to_k for l in ca_clip_layers]
    og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_clip_layers]

df = pd.read_csv('data/artists1734_prompts.csv')
artists = list(df.artist.unique())
print(len(artists))

erasure_counts = [1, 5, 10, 25, 50, 100, 500, 1000, 1500, 1734]
erasure_counts = [1, 10, 50, 100, 500, 1000, 1500, 1734]
#erasure_counts = [1500, 1734]
for count in erasure_counts:

    erasure_idxs = np.arange(len(artists))
    random.shuffle(erasure_idxs)
    erasure_idxs = erasure_idxs[:count]

    erase_artists = []
    preserve_artists = []
    for i, artist in enumerate(artists):
        if i in erasure_idxs:
            erase_artists.append(artist)
        else:
            preserve_artists.append(artist)
    print(f'Erasing {count} artists')
    with open(f'data/erase{count}.txt', 'w') as fp:
        for item in erase_artists:
            fp.write(item+'\n')


    ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
    tokenizer = ldm_stable.tokenizer

    ### get layers
    ca_layers = []
    def append_ca(net_):
        if net_.__class__.__name__ == 'CrossAttention':
            ca_layers.append(net_)
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                append_ca(net__)

    sub_nets = ldm_stable.unet.named_children()
    for net in sub_nets:
            if "down" in net[0]:
                append_ca(net[1])
            elif "up" in net[0]:
                append_ca(net[1])
            elif "mid" in net[0]:
                append_ca(net[1])

    ### get projection matrices
    ca_clip_layers = [l for l in ca_layers if l.to_v.in_features == 768]
    projection_matrices = [l.to_v for l in ca_clip_layers]
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_clip_layers]
    if with_to_k:
        projection_matrices = projection_matrices + [l.to_k for l in ca_clip_layers]
        og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_clip_layers]

    ######### EDIT THE MODEL
    #edit_model(old_text_=erase_artists, new_text_='', retain_text_= preserve_artists, lamb=0.5)
    edit_model(old_text_=erase_artists, new_text_='', retain_text_= None, lamb=0.5)
    torch.save(ldm_stable.unet.state_dict(), f'models/diffusers-erasing-{count}-with-preservation.pt')
