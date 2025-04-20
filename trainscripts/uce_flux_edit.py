import torch
torch.set_grad_enabled(False)
import argparse
import os
import copy
import time
import gc

from safetensors.torch import save_file
from diffusers import DiffusionPipeline


def UCE(model_id, erase_concepts, guide_concepts, preserve_concepts, erase_scale, preserve_scale, lamb, save_dir, exp_name, torch_dtype, device, max_sequence_length):
    # Prepare the cross attention weights required to do UCE
    pipe = DiffusionPipeline.from_pretrained(model_id, 
                                             vae=None,
                                             text_encoder=None,
                                             text_encoder_2=None,
                                             tokenizer=None,
                                             tokenizer_2=None,
                                             torch_dtype=torch_dtype, 
                                             safety_checker=None).to(device)
    uce_modules = []
    uce_module_names = []
    for name, module in pipe.transformer.named_modules():
        if 'context_embedder' in name or 'text_embedder.linear_1' in name:
            uce_modules.append(module)
            uce_module_names.append(name)
    original_modules = copy.deepcopy(uce_modules)
    uce_modules = copy.deepcopy(uce_modules)
    
    
    pipe=None
    torch.cuda.empty_cache()
    gc.collect()

    pipe = DiffusionPipeline.from_pretrained(model_id, 
                                             vae=None,
                                             transformer=None,
                                             torch_dtype=torch_dtype, 
                                             safety_checker=None).to(device)
    start_time = time.time()
    # collect text embeddings for erase concept and retain concepts
    uce_erase_embeds = {}
    for e in erase_concepts + guide_concepts + preserve_concepts:
        if e in uce_erase_embeds:
            continue
        t_emb = pipe.encode_prompt(prompt=e,
                                   prompt_2=None,
                                   device=device,
                                   num_images_per_prompt=1,
                                   max_sequence_length=max_sequence_length
                                   )
    
        last_token_idx = (pipe.tokenizer_2(e,
                                           padding="max_length",
                                           max_length=max_sequence_length,
                                           return_overflowing_tokens=False,
                                           truncation=True,
                                           return_length=False,
                                           return_tensors="pt",
                                          )['attention_mask']).sum()-2
        
    
        uce_erase_embeds[e] = [t_emb[0][:,last_token_idx,:], t_emb[1]]
    
    # collect cross attention outputs for guide concepts and retain concepts (this is for original model weights)
    uce_guide_outputs = {}
    for g in guide_concepts + preserve_concepts:
        if g in uce_guide_outputs:
            continue
    
        t_emb = uce_erase_embeds[g]
        
        for module in original_modules:
            try:
                uce_guide_outputs[g] = uce_guide_outputs.get(g, []) + [module(t_emb[0])]
            except:
                uce_guide_outputs[g] = uce_guide_outputs.get(g, []) + [module(t_emb[1])]

    ###### UCE Algorithm (variables are named according to the paper: https://arxiv.org/abs/2308.14761)
    for module_idx, module in enumerate(original_modules):
        # get original weight of the model
        w_old = module.weight
        
        mat1 = lamb * w_old
        mat2 = lamb * torch.eye(w_old.shape[1], device = device, dtype=torch_dtype)  
    
        emb_idx = 0
        if mat1.shape[-1] == 768:
            emb_idx = 1
        # Erase Concepts
        for erase_concept, guide_concept in zip(erase_concepts, guide_concepts):
            c_i = uce_erase_embeds[erase_concept][emb_idx].T
            v_i_star = uce_guide_outputs[guide_concept][module_idx].T
    
            mat1 += erase_scale * (v_i_star @ c_i.T)
            mat2 += erase_scale * (c_i @ c_i.T)
    
        # Preserve Concepts
        for preserve_concept in preserve_concepts:
            c_i = uce_erase_embeds[preserve_concept][emb_idx].T
            v_i_star = uce_guide_outputs[preserve_concept][module_idx].T
    
            mat1 += preserve_scale * (v_i_star @ c_i.T)
            mat2 += preserve_scale * (c_i @ c_i.T)
    
    
        uce_modules[module_idx].weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2.float()).to(torch_dtype))

    # save the weights
    uce_state_dict = {}
    for name, parameter in zip(uce_module_names, uce_modules):
        uce_state_dict[name+'.weight'] = parameter.weight
    save_file(uce_state_dict, os.path.join(save_dir, exp_name+'.safetensors'))
    end_time = time.time()
    print(f'\n\nErased concepts using UCE\nModel edited in {end_time-start_time} seconds\n')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainUCE',
                    description = 'UCE for erasing concepts in Stable Diffusion')
    parser.add_argument('--erase_concepts', help='prompts corresponding to concepts to erase separated by ;', type=str, required=True)
    parser.add_argument('--guide_concepts', help='Concepts to guide the erased concepts towards seperated by ;', type=str, default=None)
    parser.add_argument('--preserve_concepts', help='Concepts to preserve seperated by ;', type=str, default=None)
    parser.add_argument('--concept_type', help='type of concept being erased', choices=['art', 'object'], type=str, required=True)
    
    parser.add_argument('--model_id', help='Model to run UCE on', type=str, default="black-forest-labs/FLUX.1-schnell",)
    parser.add_argument('--device', help='cuda devices to train on', type=str, required=False, default='cuda:0')
    
    parser.add_argument('--erase_scale', help='scale to erase concepts', type=float, required=False, default=1)
    parser.add_argument('--preserve_scale', help='scale to preserve concepts', type=float, required=False, default=1)
    parser.add_argument('--lamb', help='lambda regularization term for UCE', type=float, required=False, default=0.5)
    
    parser.add_argument('--expand_prompts', help='do you wish to expand your prompts?', choices=['true', 'false'], type=str, required=False, default='false')
    
    parser.add_argument('--save_dir', help='where to save your uce model weights', type=str, default='../uce_models')
    parser.add_argument('--exp_name', help='Use this to name your saved filename', type=str, default=None)
    
    args = parser.parse_args()
    
    device = args.device
    torch_dtype = torch.float32
    model_id = args.model_id
    
    preserve_scale = args.preserve_scale
    erase_scale = args.erase_scale
    lamb = args.lamb
    
    concept_type = args.concept_type
    expand_prompts = args.expand_prompts
    
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    exp_name = args.exp_name
    if exp_name is None:
        exp_name = 'uce_test'

    max_sequence_length=512
    if 'schnell' in model_id:
        max_sequence_length=256
    # erase concepts
    erase_concepts = [concept.strip() for concept in args.erase_concepts.split(';')]
    # guide concepts
    guide_concepts = args.guide_concepts 
    if guide_concepts is None:
        guide_concepts = ''
        if concept_type == 'art':
            guide_concepts = 'art'
    guide_concepts = [concept.strip() for concept in guide_concepts.split(';')]
    if len(guide_concepts) == 1:
        guide_concepts = guide_concepts*len(erase_concepts)
    if len(guide_concepts) != len(erase_concepts):
        raise Exception('Error! The length of erase concepts and their corresponding guide concepts do not match. Please make sure they are seperated by ; and are of equal sizes')

    # preserve concepts
    if args.preserve_concepts is None:
        preserve_concepts = []
    else:
        preserve_concepts = [concept.strip() for concept in args.preserve_concepts.split(';')]
    
    

    if expand_prompts == 'true':
        erase_concepts_ = copy.deepcopy(erase_concepts)
        guide_concepts_ = copy.deepcopy(guide_concepts)

        for concept, guide_concept in zip(erase_concepts_, guide_concepts_):
            if concept_type == 'art':
                erase_concepts.extend([f'painting by {concept}',
                                       f'art by {concept}',
                                       f'artwork by {concept}',
                                       f'picture by {concept}',
                                       f'style of {concept}'
                                      ]
                                     )
            elif concept_type=='object':
                erase_concepts.extend([f'image of {concept}',
                                       f'photo of {concept}',
                                       f'portrait of {concept}',
                                       f'picture of {concept}',
                                       f'painting of {concept}'
                                      ]
                                     )
                
            guide_concepts.extend([guide_concept]*5)


    print(f"\n\nErasing: {erase_concepts}\n")
    print(f"Guiding: {guide_concepts}\n")
    print(f"Preserving: {preserve_concepts}\n")
    
    UCE(model_id, erase_concepts, guide_concepts, preserve_concepts, erase_scale, preserve_scale, lamb, save_dir, exp_name, torch_dtype, device, max_sequence_length)
    