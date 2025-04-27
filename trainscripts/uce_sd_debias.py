import torch
torch.set_grad_enabled(False)
import argparse
import os
import copy
import time
import numpy as np
from tqdm.auto import tqdm

from safetensors.torch import save_file
from diffusers import DiffusionPipeline
from transformers import pipeline

def get_ratios(pipe, clip, uce_module_names, uce_modules, edit_concepts, debias_concepts, desired_ratios, max_diff, step_size =0.1, num_images_per_prompt=10,num_inference_steps=20, guidance_scale=7.5):
    uce_state_dict = {}
    for name, parameter in zip(uce_module_names, uce_modules):
        uce_state_dict[name+'.weight'] = parameter.weight

    pipe.unet.load_state_dict(uce_state_dict, strict=False)
    direction_scale = [] 
    for concept in edit_concepts:
        images = pipe(concept,
                     num_inference_steps=num_inference_steps,
                     num_images_per_prompt=num_images_per_prompt,
                      guidance_scale=guidance_scale
                     ).images
        results = clip(images, candidate_labels=debias_concepts)
        results = np.array([r[0]['label'] for r in results])
        
        ratios = np.array([desired - (sum(results==c)/len(results)) for c, desired in zip(debias_concepts, desired_ratios)])
        if max(ratios) < max_diff and abs(min(ratios)) < max_diff:
            ratios = 0 * ratios
            
        direction_scale.append(ratios)
    return np.array(direction_scale)
    
def UCE(pipe, clip, edit_concepts, debias_concepts, preserve_concepts, edit_scale, preserve_scale, lamb, save_dir, exp_name, max_diff, step_size,num_images_per_prompt, num_inference_steps, guidance_scale):
    # Prepare the cross attention weights required to do UCE
    uce_modules = []
    uce_module_names = []
    for name, module in pipe.unet.named_modules():
        if 'attn2' in name and (name.endswith('to_v') or name.endswith('to_k')):
            uce_modules.append(module)
            uce_module_names.append(name)
    original_modules = copy.deepcopy(uce_modules)
    uce_modules = copy.deepcopy(uce_modules)

    # collect text embeddings for edit concept and preserve concepts
    uce_erase_embeds = {}
    for e in edit_concepts + debias_concepts + preserve_concepts:
        if e in uce_erase_embeds:
            continue
        t_emb = pipe.encode_prompt(prompt=e,
                                   device=device,
                                   num_images_per_prompt=1,
                                   do_classifier_free_guidance=False)
    
        last_token_idx = (pipe.tokenizer(e,
                                          padding="max_length",
                                          max_length=pipe.tokenizer.model_max_length,
                                          truncation=True,
                                          return_tensors="pt",
                                         )['attention_mask']).sum()-2
    
    
        uce_erase_embeds[e] = t_emb[0][:,last_token_idx,:]
    
    uce_guide_outputs = {}
    for g in edit_concepts + debias_concepts + preserve_concepts:
        if g in uce_guide_outputs:
            continue
    
        t_emb = pipe.encode_prompt(prompt=g,
                                   device=device,
                                   num_images_per_prompt=1,
                                   do_classifier_free_guidance=False)
    
        last_token_idx = (pipe.tokenizer(g,
                                          padding="max_length",
                                          max_length=pipe.tokenizer.model_max_length,
                                          truncation=True,
                                          return_tensors="pt",
                                         )['attention_mask']).sum()-2
    
        t_emb = t_emb[0][:,last_token_idx,:]
        
        for module in original_modules:
            uce_guide_outputs[g] = uce_guide_outputs.get(g, []) + [module(t_emb)]
    
    pipe = pipe.to(torch.bfloat16)

    ###### UCE Algorithm (variables are named according to the paper: https://arxiv.org/abs/2308.14761)
    start_time = time.time()
    pbar = tqdm(range(max_iterations), desc="UCE Debiasing")
    for iteration in range(max_iterations):
        direction_scale = get_ratios(pipe=pipe, 
                                     clip=clip, 
                                     uce_module_names=uce_module_names, 
                                     uce_modules=uce_modules, 
                                     edit_concepts=edit_concepts,
                                     debias_concepts=debias_concepts, 
                                     desired_ratios=desired_ratios,
                                     max_diff=max_diff,
                                     step_size=step_size, 
                                     num_images_per_prompt=num_images_per_prompt,
                                     num_inference_steps=num_inference_steps, 
                                     guidance_scale=guidance_scale)
    
        pbar.set_postfix(ratio_diff=direction_scale) 
        if np.abs(direction_scale).max() == 0:
            print("All concepts are debiased")
            break
        
        for module_idx, module in enumerate(original_modules):
            # get original weight of the model
            w_old = module.weight
            
            mat1 = lamb * w_old
            mat2 = lamb * torch.eye(w_old.shape[1], device = device, dtype=torch_dtype)  
        
            # Edit Concepts
            for idx, edit_concept in enumerate(edit_concepts):
                c_i = uce_erase_embeds[edit_concept].T
                v_i_star = uce_guide_outputs[edit_concept][module_idx]
                for i, concept in enumerate(debias_concepts):
                    v_i_star += direction_scale[idx][i]*uce_guide_outputs[concept][module_idx]
                v_i_star = v_i_star.T
        
                mat1 += edit_scale * (v_i_star @ c_i.T)
                mat2 += edit_scale * (c_i @ c_i.T)
        
            # preserve Concepts
            for preserve_concept in preserve_concepts:
                c_i = uce_erase_embeds[preserve_concept].T
                v_i_star = uce_guide_outputs[preserve_concept][module_idx].T
        
                mat1 += preserve_scale * (v_i_star @ c_i.T)
                mat2 += preserve_scale * (c_i @ c_i.T)
        
            uce_modules[module_idx].weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2.float()).to(torch_dtype))
        pbar.update(1)
    end_time = time.time()
    # save the weights
    uce_state_dict = {}
    for name, parameter in zip(uce_module_names, uce_modules):
        uce_state_dict[name+'.weight'] = parameter.weight
    save_file(uce_state_dict, os.path.join(save_dir, exp_name+'.safetensors')) 

    print(f'\n\nDebiased concepts using UCE\nModel edited in {end_time-start_time} seconds\n')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainUCE',
                    description = 'UCE for erasing concepts in Stable Diffusion')
    parser.add_argument('--edit_concepts', help='prompts corresponding to concepts to edit separated by ;', type=str, required=True)
    parser.add_argument('--debias_concepts', help='Concepts to debias the edit concepts towards seperated by ;', type=str, default=None)
    parser.add_argument('--preserve_concepts', help='Concepts to preserve seperated by ;', type=str, default=None)
    
    parser.add_argument('--model_id', help='Model to run UCE on', type=str, default="CompVis/stable-diffusion-v1-4",)
    parser.add_argument('--device', help='cuda devices to train on', type=str, required=False, default='cuda:0')
    
    parser.add_argument('--edit_scale', help='scale to edit concepts', type=float, required=False, default=1)
    parser.add_argument('--preserve_scale', help='scale to preserve concepts', type=float, required=False, default=1)
    parser.add_argument('--lamb', help='lambda regularization term for UCE', type=float, required=False, default=0.5)
    
    parser.add_argument('--save_dir', help='where to save your uce model weights', type=str, default='../uce_models')
    parser.add_argument('--exp_name', help='Use this to name your saved filename', type=str, default=None)

    parser.add_argument('--desired_ratios', type=float, nargs='+', 
                        default=[0.5, 0.5], 
                        help='List of desired ratios for debiasing concepts(default: [0.5, 0.5])')
    
    parser.add_argument('--max_iterations', type=int, 
                        default=30, 
                        help='Maximum number of iterations to debias using UCE(default: 30)')
    
    parser.add_argument('--max_diff', type=float, 
                        default=0.05, 
                        help='Maximum difference allowed as error from desired ratio (default: 0.05)')
    
    parser.add_argument('--step_size', type=float, 
                        default=0.1, 
                        help='Step size for v* updates(default: 0.1)')
    
    parser.add_argument('--num_images_per_prompt', type=int, 
                        default=10, 
                        help='Number of images per prompt (default: 10)')
    
    parser.add_argument('--num_inference_steps', type=int, 
                        default=20, 
                        help='Number of inference steps (default: 20)')
    
    parser.add_argument('--guidance_scale', type=float, 
                        default=7.5, 
                        help='Guidance scale (default: 7.5)')
    
    args = parser.parse_args()
    
    device = args.device
    torch_dtype = torch.float32
    model_id = args.model_id
    
    preserve_scale = args.preserve_scale
    edit_scale = args.edit_scale
    lamb = args.lamb
    
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    exp_name = args.exp_name
    if exp_name is None:
        exp_name = 'uce_test'

    desired_ratios = args.desired_ratios
    max_iterations = args.max_iterations
    max_diff = args.max_diff
    step_size = args.step_size
    num_images_per_prompt = args.num_images_per_prompt
    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale

    # edit concepts
    edit_concepts = [concept.strip() for concept in args.edit_concepts.split(';')]
    # debias concepts
    debias_concepts = [concept.strip() for concept in args.debias_concepts.split(';')]
    
    if len(debias_concepts) != len(desired_ratios):
        raise Exception('Error! The length of debias concepts and their corresponding desired ratios concepts do not match.')

    # preserve concepts
    if args.preserve_concepts is None:
        preserve_concepts = []
    else:
        preserve_concepts = [concept.strip() for concept in args.preserve_concepts.split(';')]
    

    print(f"\n\nEditing: {edit_concepts}\n")
    print(f"Debias Across: {debias_concepts}\n")
    print(f"Preserving: {preserve_concepts}\n")
    
    pipe = DiffusionPipeline.from_pretrained(model_id, 
                                             torch_dtype=torch_dtype, 
                                             safety_checker=None).to(device)
    pipe.set_progress_bar_config(disable=True)

    clip = pipeline(
       task="zero-shot-image-classification",
       model="openai/clip-vit-base-patch32",
       torch_dtype=torch.bfloat16,
       device=0
    )
    
    UCE(pipe, clip, edit_concepts, debias_concepts, preserve_concepts, edit_scale, preserve_scale, lamb, save_dir, exp_name, max_diff, step_size,num_images_per_prompt, num_inference_steps, guidance_scale)