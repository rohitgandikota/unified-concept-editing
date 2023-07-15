import numpy as np
import torch
import random
import pandas as pd
from PIL import Image
import pandas as pd 
import argparse
import requests
import os, glob
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
import abc
import copy

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


def generate_for_text(ldm_stable, test_text, num_samples = 9, seed = 1231):
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    images = text2image_ldm_stable(ldm_stable, [test_text]*num_samples, latent=None, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=g, low_resource=LOW_RESOURCE)
    return view_images(images)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_ratios(ldm_stable, prev_ratio, ratio_diff, max_ratio_gap, concepts, classes, num_samples=10, num_loops=3):
    seeds = np.random.randint(5000,size=5) 
    ratios = []
    for idx, concept in enumerate(concepts):
        if ratio_diff is not None:
            if ratio_diff[idx] < max_ratio_gap:
                print(f'Bypassing Concept {idx+1}')
                ratios.append(prev_ratio[idx])
                continue
        prompt = f'{concept}'
        probs_full = []
        test_prompts = [f'{class_}' for class_ in classes[idx]]
        with torch.no_grad():
            for seed in seeds:
    #             if i == num_loops:
    #                 break
                g = torch.Generator(device='cpu')
                g.manual_seed(int(seed))
                images = ldm_stable(prompt,num_images_per_prompt=num_samples, num_inference_steps=20, generator = g).images

                inputs = clip_processor(text=test_prompts, images=images, return_tensors="pt", padding=True)

                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
                tmax = probs.max(1, keepdim=True)[0]
                mask = probs.ge(tmax)
                probs_full.append(mask.float())
                
        ratios.append(torch.cat(probs_full).mean(axis=0))
#     male = float(probs[0][0])
    return ratios

## get arguments for our script
with_to_k = False
with_augs = True
train_func = "train_closed_form"

### load model
LOW_RESOURCE = False
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77

def edit_model(ldm_stable, old_text_, new_text_, retain_text_, add=True, layers_to_edit=None, lamb=0.1, erase_scale = 0.1, preserve_scale = 0.1, with_to_k=True, num_images=1):
    ### collect all the cross attns modules
    max_bias_diff = 0.05
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

    print(old_texts, new_texts)
    desired_ratios = [torch.ones(len(c))/len(c) for c in new_texts ]
    weight_step = 0.1
    weights = [torch.zeros(len(c)) for c in new_texts ]
    #################################### START OUTER LOOP #########################
    for i in range(30):
        max_ratio_gap = max_bias_diff
        if i == 0:
            prev_ratio = None
            ratio_diff = None
        else:
            prev_ratio = ratios
            ratio_diff = max_change
        ratios = [0 for _ in desired_ratios]
        ratios = get_ratios(ldm_stable=ldm_stable, prev_ratio = prev_ratio, ratio_diff=ratio_diff, max_ratio_gap=max_ratio_gap, concepts=old_texts, classes=new_texts, num_samples= num_images)
        if i == 0 :
            init_ratios = ratios
        print(ratios)
        max_change = [(ratio - desired_ratio).abs().max() for ratio, desired_ratio in zip(ratios,desired_ratios)]


        if max(max_change) < max_bias_diff:
            print(f'All concepts are debiased at Iteration:{i}')
            break
         #### restart LDM parameters
#         num_ca_clip_layers = len(ca_layers)
#         for idx_, l in enumerate(ca_layers):
#             l.to_v = copy.deepcopy(og_matrices[idx_])
#             projection_matrices[idx_] = l.to_v
#             if with_to_k:
#                 l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
#                 projection_matrices[num_ca_clip_layers + idx_] = l.to_k
        
        weights_delta = [weight_step * (desired_ratio - ratio) for ratio, desired_ratio in zip(ratios, desired_ratios)]
        weights_delta = [weights_delta[idx] if max_c>max_bias_diff else weights_delta[idx]*0 for idx, max_c in enumerate(max_change)]
        
        # check if the ratio is attained. If so, add it to preservation and skip the ratios check again
        ret_text_add = [old_texts[idx] for idx, weight in enumerate(weights_delta) if weight[0]==0]
        if len(ret_text_add)>0:
            ret_texts = ret_texts+ret_text_add
            ret_texts = list(np.unique(ret_texts))
        weights = weights_delta
#         weights = [weight + weights_delta[idx] for idx, weight in enumerate(weights)]
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

                for cnt, t in enumerate(zip(old_texts, new_texts)):
                    old_text = t[0]
                    new_text = t[1]
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
                    final_token_idx = text_input.attention_mask[0].sum().item()-2
                    final_token_idx_new = [text_input.attention_mask[i].sum().item()-2 for i in range(1, len(text_input.attention_mask))]
                    farthest = max(final_token_idx_new+[final_token_idx])
                    new_emb = text_embeddings[1:]

                    context = old_emb.detach()[final_token_idx:len(old_emb)-max(0,farthest-final_token_idx)]
                    values = []
                    with torch.no_grad():
                        for layer in projection_matrices:
                            o_embs = layer(old_emb).detach()
                            o_embs = o_embs[final_token_idx:len(o_embs)-max(0,farthest-final_token_idx)]
#                             print(f'O_EMBS: {final_token_idx}-{len(o_embs)-max(0,farthest-final_token_idx)}')
                            embs = layer(new_emb[:]).detach()
                            target = o_embs
                            for j, emb in enumerate(embs):
                                u = emb
                                u = u[final_token_idx_new[j]:len(u)-max(0,farthest-final_token_idx_new[j])]
#                                 print(f'U_{j}: {final_token_idx_new[j]}-{len(u)-max(0,farthest-final_token_idx_new[j])}')
                                u = u / u.norm()
                                o_emb_proj = (u*o_embs).sum()
                                target += (weights[cnt][j]*o_embs.norm())*u 
                            values.append(target.detach())    
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainUSD',
                    description = 'Finetuning stable diffusion to debias the concepts')
    parser.add_argument('--concept', help='prompt corresponding to concept to erase', type=str, required=True)
    parser.add_argument('--attributes', help='Attributes to debias', type=str, default='male, female')
    parser.add_argument('--device', help='cuda devices to train on', type=str, required=False, default='0')
    parser.add_argument('--base', help='base version for stable diffusion', type=str, required=False, default='1.4')
    parser.add_argument('--num_images', help='number of images per concept to generate for validation', type=int, required=False, default=10)
    args = parser.parse_args()

    device = f'cuda:{args.device}'
    num_images = args.num_images
    concepts = args.concept.split(',')
    concepts = [con.strip() for con in concepts]
    print_text=''
    for txt in concepts:
        print_text += txt.lower()+'_'
    if concepts[0] == 'default0':
        print('Using default training setup')
        concepts = ['attendant', 'cashier', 'teacher','nurse', 'assistant', 'secretary', 'cleaner', 'receptionist','clerk','counselor', 'designer', 'hairdresser', 'writer', 'housekeeper', 'baker', 'librarian','tailor','driver','supervisor', 'janitor', 'cook', 'laborer', 'construction worker', 'developer', 'carpenter','manager', 'lawyer', 'farmer', 'salesperson', 'physician', 'guard', 'analyst', 'mechanic', 'sheriff', 'CEO', 'doctor', 'chef']
    old_texts = []
    concepts_ = []
    for concept in concepts:
        old_texts.append(f'image of {concept}')
        old_texts.append(f'photo of {concept}')
        old_texts.append(f'portrait of {concept}')
        old_texts.append(f'picture of {concept}')
        old_texts.append(f'{concept}')
        concepts_.extend([concept]*5)
    attributes = args.attributes
    attributes = attributes.split(',')
    attributes = [att.strip() for att in attributes]
    
    print_text  = print_text[:-1] + '-attributes-'
    for txt in attributes:
        print_text += txt.lower().replace(' ','9')+'_'
    print_text  = print_text[:-1]
    
    new_texts = [[text.replace(concepts_[idx], att) for att in attributes] for idx, text in enumerate(old_texts) ]
    
    df = pd.read_csv('data/profession_prompts.csv')

    retain_texts = list(df.profession.unique())
    ### default concepts to erase
  
    old_texts_lower = [text.lower() for text in old_texts]
    retain_texts = [text for text in retain_texts if text.lower() not in old_texts_lower]
    sd14="CompVis/stable-diffusion-v1-4"
    sd21='stabilityai/stable-diffusion-2-1-base'
    if args.base=='1.4':
        model_version = sd14
    elif args.base=='2.1':
        model_version = sd21
    else:
        model_version = sd14
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_version).to(device)
    print_text += f"-sd_{args.base.replace('.','_')}" 
    print(print_text)
    ldm_stable, weights, init_ratios, final_ratios = edit_model(ldm_stable= ldm_stable, old_text_= old_texts, new_text_=new_texts, add=False, retain_text_= retain_texts, lamb=0.5, erase_scale = 1, preserve_scale = .1, num_images=num_images)
    
    torch.save(ldm_stable.unet.state_dict(), f'models/unbiased-{print_text}.pt')

    with open(f'data/unbiased-{print_text}.txt', 'w') as fp:
        fp.write(str(old_texts)+'\n'+str(weights)+'\n'+str(init_ratios)+'\n'+str(final_ratios))