from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import argparse
import os
import numpy as np

def get_A(z_i, z_j):
    z_i = z_i[:, None]
    z_j = z_j[:, None]
    return (np.matmul(z_i, z_i.T) + np.matmul(z_j, z_j.T) - np.matmul(z_i, z_j.T) - np.matmul(z_j, z_i.T))


def get_M(embeddings, S):
    d = embeddings.shape[1]
    M = np.zeros((d, d))
    for s in S:
        M  += get_A(embeddings[s[0]], embeddings[s[1]])
    return M / len(S)

def generate_images(model_name, prompts_path, debias_concepts, save_path, device='cuda:0', guidance_scale = 7.5, image_size=512, ddim_steps=100, num_samples=10, from_case=0, till_case=1000000, base='1.4'):
    
    if base == '1.4':
        model_version = "CompVis/stable-diffusion-v1-4"
    elif base == '2.1':
        model_version = 'stabilityai/stable-diffusion-2-1-base'
    else:
        model_version = "CompVis/stable-diffusion-v1-4"
    vae = AutoencoderKL.from_pretrained(model_version, subfolder="vae")

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained(model_version, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_version, subfolder="text_encoder")

    # 3. The UNet model for generating the latents.
#     name = 'compvis-word_VanGogh-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05'
    unet = UNet2DConditionModel.from_pretrained(model_version, subfolder="unet")
    if model_name != 'original':
        model_path = f'models/{model_name}'
        unet.load_state_dict(torch.load(model_path))
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    torch_device = device
    df = pd.read_csv(prompts_path)
    prompts = df.prompt
    seeds = df.evaluation_seed
    case_numbers = df.case_number

    folder_path = f'{save_path}/{model_name.replace("diffusers-","").replace(".pt","")}'
    os.makedirs(folder_path, exist_ok=True)
    
    # Construct Positive Pair
    candidate_prompt = []
    S = []
    counter = 0
    for train_cls_i in debias_concepts:
        train_cls_i = train_cls_i.lower()
        candidate_prompt += ['A photo of a male {}.'.format(train_cls_i), 'A photo of a female {}.'.format(train_cls_i)]
        S += [[counter, counter + 1]]
        counter += 2

    candidate_input = tokenizer(candidate_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        candidate_embeddings = text_encoder(candidate_input.input_ids.to(torch_device))[0]#.cpu().numpy()
    candidate_embeddings = candidate_embeddings[torch.arange(candidate_embeddings.shape[0]), candidate_input['input_ids'].argmax(-1)]
    candidate_embeddings = F.normalize(candidate_embeddings, dim=-1).cpu().numpy()

    # Compute Calibration Matrix
    M =  get_M(candidate_embeddings, S)
    G = 500.0 * M + np.eye(M.shape[0]) 
    P = np.linalg.inv(G)
    P = torch.tensor(P).to(device)


    for _, row in df.iterrows():
        prompt = [str(row.prompt)]*num_samples
        seed = row.evaluation_seed
        case_number = row.case_number
        if not (case_number>=from_case and case_number<=till_case):
            continue

        height = image_size                        # default height of Stable Diffusion
        width = image_size                         # default width of Stable Diffusion

        num_inference_steps = ddim_steps           # Number of denoising steps

        guidance_scale = guidance_scale            # Scale for classifier-free guidance

        generator = torch.manual_seed(seed)    # Seed generator to create the inital latent noise

        batch_size = len(prompt)

        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
        
        # Debias Text Embedding
        text_embeddings = torch.matmul(text_embeddings, P.T.float())
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(torch_device)

        scheduler.set_timesteps(num_inference_steps)

        latents = latents * scheduler.init_noise_sigma

        from tqdm.auto import tqdm

        scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        for num, im in enumerate(pil_images):
            im.save(f"{folder_path}/{case_number}_{num}.png")

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImages',
                    description = 'Generate Images using Diffusers Code')
    parser.add_argument('--model_name', help='name of model', type=str, required=True)
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=True)
    parser.add_argument('--debias_concepts', help='Concepts to debias', type=str, required=False, default = '')
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=True)
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--base', help='version of stable diffusion to run', type=str, required=False, default='2.1')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--till_case', help='continue generating from case_number', type=int, required=False, default=1000000)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--num_samples', help='number of samples per prompt', type=int, required=False, default=1)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=100)
    args = parser.parse_args()
    
    model_name = args.model_name
    prompts_path = args.prompts_path
    save_path = args.save_path
    debias_concepts = args.debias_concepts.split(',')
    debias_concepts = [b.strip() for b in debias_concepts]
    if debias_concepts == ['']:
        print('Using default train list')
        debias_concepts = ['Actor', 'Architect', 'Audiologist', 'Author', 'Baker', 'Barber', 'Blacksmith', 'Bricklayer', 'Bus Driver', 'Butcher', 'Chef', 'Chemist', 'Cleaner', 'Coach', 'Comedian', 'Computer Programmer', 'Construction Worker', 'Consultant', 'Counselor', 'Dancer', 'Dentist', 'Designer', 'Dietitian', 'DJ', 'Doctor', 'Driver', 'Economist', 'Electrician', 'Engineer', 'Entrepreneur', 'Farmer', 'Florist', 'Graphic Designer', 'Hairdresser', 'Historian', 'Journalist', 'Judge', 'Lawyer', 'Librarian', 'Magician', 'Makeup Artist', 'Mathematician', 'Marine Biologist', 'Mechanic', 'Model', 'Musician', 'Nanny', 'Nurse', 'Optician', 'Painter', 'Pastry Chef', 'Pediatrician', 'Photographer', 'Plumber', 'Police Officer', 'Politician', 'Professor', 'Psychologist', 'Real Estate Agent', 'Receptionist', 'Recruiter', 'Researcher', 'Sailor', 'Salesperson', 'Surveyor', 'Singer', 'Social Worker', 'Software Developer', 'Statistician', 'Surgeon', 'Teacher', 'Technician', 'Therapist', 'Tour Guide', 'Translator', 'Vet', 'Videographer', 'Waiter', 'Writer', 'Zoologist']
    device = args.device
    guidance_scale = args.guidance_scale
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    num_samples= args.num_samples
    from_case = args.from_case
    till_case = args.till_case
    generate_images(model_name=model_name, prompts_path=prompts_path,debias_concepts=debias_concepts, save_path=save_path, device=device,
                    guidance_scale = guidance_scale, image_size=image_size, ddim_steps=ddim_steps, num_samples=num_samples,from_case=from_case, till_case=till_case, base=args.base)
