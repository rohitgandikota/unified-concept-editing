from diffusers import DiffusionPipeline
import torch
from PIL import Image
import pandas as pd
import argparse
import os
torch.set_grad_enabled(False)
from safetensors.torch import load_file

def generate_images(model_id, uce_model_path, prompts_path, save_path, exp_name='test', device='cuda:0', torch_dtype=torch.bfloat16, guidance_scale = 7.5, num_inference_steps=100, num_images_per_prompt=10, from_case=0, till_case=1000000):
    
    # 1. Load the pipe
    pipe = DiffusionPipeline.from_pretrained(model_id, 
                                         torch_dtype=torch_dtype, 
                                         safety_checker=None).to(device)

    if uce_model_path is not None:
        uce_weights = load_file(uce_model_path)
        pipe.unet.load_state_dict(uce_weights, strict=False)
    
    df = pd.read_csv(prompts_path)
    prompts = df.prompt
    seeds = df.evaluation_seed
    case_numbers = df.case_number

    folder_path = f'{save_path}/{exp_name}'
    os.makedirs(folder_path, exist_ok=True)

    for _, row in df.iterrows():
        prompt = str(row.prompt)
        seed = row.evaluation_seed
        case_number = row.case_number
        if not (case_number>=from_case and case_number<=till_case):
            continue

        
        pil_images = pipe(prompt=prompt,
                          num_inference_steps=num_inference_steps,
                          guidance_scale=guidance_scale,
                          num_images_per_prompt=num_images_per_prompt,
                          generator=torch.Generator().manual_seed(seed)
                         ).images
                          
                          
        for num, im in enumerate(pil_images):
            im.save(f"{folder_path}/{case_number}_{num}.png")

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImages',
                    description = 'Generate Images using Diffusers Code')
    parser.add_argument('--model_id', help='hf repo id for the model you want to test', type=str, required=False, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--uce_model_path', help='path for uce model', type=str, required=False, default=None)
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=True)
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=False, default='../uce_results/')
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--exp_name', help='foldername to save the results', type=str, required=False, default='test_images')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--till_case', help='continue generating from case_number', type=int, required=False, default=1000000)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--num_images_per_prompt', help='number of samples per prompt', type=int, required=False, default=1)
    parser.add_argument('--num_inference_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    args = parser.parse_args()
    
    model_id = args.model_id
    uce_model_path = args.uce_model_path
    prompts_path = args.prompts_path
    save_path = args.save_path
    device = args.device
    guidance_scale = args.guidance_scale
    exp_name = args.exp_name
    num_images_per_prompt= args.num_images_per_prompt
    from_case = args.from_case
    till_case = args.till_case
    num_inference_steps = args.num_inference_steps
    generate_images(model_id=model_id, uce_model_path=uce_model_path, prompts_path=prompts_path, save_path=save_path, exp_name=exp_name, device=device, torch_dtype=torch.bfloat16, guidance_scale = guidance_scale, num_inference_steps=num_inference_steps, num_images_per_prompt=num_images_per_prompt, from_case=from_case, till_case=till_case)
