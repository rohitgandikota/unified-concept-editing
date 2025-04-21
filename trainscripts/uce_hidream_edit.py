import torch
torch.set_grad_enabled(False)
import argparse
import os
import copy
import time
import gc

from safetensors.torch import save_file
from diffusers import DiffusionPipeline
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from diffusers import UniPCMultistepScheduler, HiDreamImagePipeline

def UCE(model_id, edit_concepts, guide_concepts, preserve_concepts, erase_scale, preserve_scale, lamb, save_dir, exp_name, torch_dtype, device, max_sequence_length):
    # Prepare the cross attention weights required to do UCE
    pipe = HiDreamImagePipeline.from_pretrained(
        model_id,
        tokenizer_4=None,
        text_encoder_4=None,
        tokenizer_3=None,
        text_encoder_3=None,
        tokenizer_2=None,
        text_encoder_2=None,
        tokenizer=None,
        text_encoder=None,
        vae=None,
        torch_dtype=torch_dtype,
    )
    
    uce_modules = []
    uce_module_names = []
    for name, module in pipe.transformer.named_modules():
        if 'caption_projection' in name and 'linear' in name:
            uce_modules.append(module.to(device))
            uce_module_names.append(name)
    original_modules = copy.deepcopy(uce_modules)
    uce_modules = copy.deepcopy(uce_modules)
    
    llama_layers = pipe.transformer.config.llama_layers
    pipe = None
    torch.cuda.empty_cache()
    gc.collect()

    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch_dtype,
    )
    pipe = HiDreamImagePipeline.from_pretrained(
        model_id,
        transformer=None,
        vae=None,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        tokenizer_3=None,
        text_encoder_3=None,
        tokenizer_2=None,
        text_encoder_2=None,
        tokenizer=None,
        text_encoder=None,
        torch_dtype=torch_dtype,
    ).to(device)
    
    
    # collect text embeddings for erase concept and retain concepts
    uce_erase_embeds = {}
    for e in edit_concepts + guide_concepts + preserve_concepts:
        if e in uce_erase_embeds:
            continue
        t_emb_llama = pipe._get_llama3_prompt_embeds(e, 
                                                     max_sequence_length, 
                                                     device, 
                                                     torch_dtype)
        

        last_token_idx_llama = (pipe.tokenizer_4(e,
                                       padding="max_length",
                                       max_length=min(max_sequence_length, pipe.tokenizer_4.model_max_length),
                                       truncation=True,
                                       add_special_tokens=True,
                                       return_tensors="pt",
                                      )['attention_mask']).sum()-2
        
        
    
        t_emb_llama = t_emb_llama[:,:,last_token_idx_llama,:]
        
        
        uce_erase_embeds[e] = [t_emb_llama[idx].to(device) for idx in llama_layers] 
    pipe = None
    tokenizer_4 = None
    text_encoder_4 = None
    torch.cuda.empty_cache()
    gc.collect()
    pipe = HiDreamImagePipeline.from_pretrained(
        model_id,
        transformer=None,
        vae=None,
        tokenizer_4=None,
        text_encoder_4=None,
        tokenizer_2=None,
        text_encoder_2=None,
        tokenizer=None,
        text_encoder=None,
        torch_dtype=torch_dtype,
    ).to(device)
    for e in edit_concepts + guide_concepts + preserve_concepts:
        t_emb_t5 = pipe._get_t5_prompt_embeds(e, 
                                              max_sequence_length,
                                              device, 
                                              torch_dtype)
        last_token_idx_t5 = (pipe.tokenizer_3(
                                            e,
                                            padding="max_length",
                                            max_length=min(max_sequence_length, pipe.tokenizer_3.model_max_length),
                                            truncation=True,
                                            add_special_tokens=True,
                                            return_tensors="pt",
                                        )['attention_mask']).sum()-2
        t_emb_t5 = t_emb_t5[:,last_token_idx_t5,:]
        uce_erase_embeds[e] += [t_emb_t5.to(device)]
    pipe = None
    torch.cuda.empty_cache()
    gc.collect()
    
    start_time = time.time()
    # collect cross attention outputs for guide concepts and retain concepts (this is for original model weights)
    uce_guide_outputs = {}
    for g in guide_concepts + preserve_concepts:
        if g in uce_guide_outputs:
            continue
    
        t_emb = uce_erase_embeds[g]
    
        # llama embeddings
        for module_idx, module in enumerate(original_modules):
            uce_guide_outputs[g] = uce_guide_outputs.get(g, []) + [module(t_emb[module_idx])]
        # t5 embeddings
        uce_guide_outputs[g] += [original_modules[-1](t_emb[-1])]
        
    ###### UCE Algorithm (variables are named according to the paper: https://arxiv.org/abs/2308.14761)
    for module_idx, module in enumerate(original_modules + original_modules[-1:]):
        if module_idx == len(original_modules):
            module_idx-=1
        # get original weight of the model
        w_old = module.weight
        
        mat1 = lamb * w_old
        mat2 = lamb * torch.eye(w_old.shape[1], device = device, dtype=torch_dtype)  
    
        # Erase Concepts
        for erase_concept, guide_concept in zip(edit_concepts, guide_concepts):
            c_i = uce_erase_embeds[erase_concept][module_idx].T
            v_i_star = uce_guide_outputs[guide_concept][module_idx].T
    
            mat1 += erase_scale * (v_i_star @ c_i.T)
            mat2 += erase_scale * (c_i @ c_i.T)
    
        # Preserve Concepts
        for preserve_concept in preserve_concepts:
            c_i = uce_erase_embeds[preserve_concept][module_idx].T
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
    parser.add_argument('--edit_concepts', help='prompts corresponding to concepts to erase separated by ;', type=str, required=True)
    parser.add_argument('--guide_concepts', help='Concepts to guide the erased concepts towards seperated by ;', type=str, default=None)
    parser.add_argument('--preserve_concepts', help='Concepts to preserve seperated by ;', type=str, default=None)
    parser.add_argument('--concept_type', help='type of concept being erased', choices=['art', 'object'], type=str, required=True)
    
    parser.add_argument('--model_id', help='Model to run UCE on', type=str, default="HiDream-ai/HiDream-I1-Full",)
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

    max_sequence_length=128
    # erase concepts
    edit_concepts = [concept.strip() for concept in args.edit_concepts.split(';')]
    # guide concepts
    guide_concepts = args.guide_concepts 
    if guide_concepts is None:
        guide_concepts = ''
        if concept_type == 'art':
            guide_concepts = 'art'
    guide_concepts = [concept.strip() for concept in guide_concepts.split(';')]
    if len(guide_concepts) == 1:
        guide_concepts = guide_concepts*len(edit_concepts)
    if len(guide_concepts) != len(edit_concepts):
        raise Exception('Error! The length of erase concepts and their corresponding guide concepts do not match. Please make sure they are seperated by ; and are of equal sizes')

    # preserve concepts
    if args.preserve_concepts is None:
        preserve_concepts = []
    else:
        preserve_concepts = [concept.strip() for concept in args.preserve_concepts.split(';')]
    
    

    if expand_prompts == 'true':
        edit_concepts_ = copy.deepcopy(edit_concepts)
        guide_concepts_ = copy.deepcopy(guide_concepts)

        for concept, guide_concept in zip(edit_concepts_, guide_concepts_):
            if concept_type == 'art':
                edit_concepts.extend([f'painting by {concept}',
                                       f'art by {concept}',
                                       f'artwork by {concept}',
                                       f'picture by {concept}',
                                       f'style of {concept}'
                                      ]
                                     )
                guide_concepts.extend([f'painting by {guide_concept}',
                                       f'art by {guide_concept}',
                                       f'artwork by {guide_concept}',
                                       f'picture by {guide_concept}',
                                       f'style of {guide_concept}'
                                      ]
                                     )

            else:
                edit_concepts.extend([f'image of {concept}',
                                       f'photo of {concept}',
                                       f'portrait of {concept}',
                                       f'picture of {concept}',
                                       f'painting of {concept}'
                                      ]
                                     )
                guide_concepts.extend([f'image of {guide_concept}',
                                       f'photo of {guide_concept}',
                                       f'portrait of {guide_concept}',
                                       f'picture of {guide_concept}',
                                       f'painting of {guide_concept}'
                                      ]
                                     )


    print(f"\n\nErasing: {edit_concepts}\n")
    print(f"Guiding: {guide_concepts}\n")
    print(f"Preserving: {preserve_concepts}\n")
    
    UCE(model_id, edit_concepts, guide_concepts, preserve_concepts, erase_scale, preserve_scale, lamb, save_dir, exp_name, torch_dtype, device, max_sequence_length)
    