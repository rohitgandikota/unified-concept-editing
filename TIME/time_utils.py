import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd


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
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm_stable(
        model,
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=None,
        latent=None,
        low_resource=False,
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
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, latents, context, t, guidance_scale, low_resource)

    image = latent2image(model.vae, latents)

    return image


def edit_closed_form(projection_matrices, contexts, valuess, layers_to_edit=None, lamb=0.1):
    layers_to_edit = ast.literal_eval(layers_to_edit) if type(layers_to_edit) == str else layers_to_edit
    lamb = ast.literal_eval(lamb) if type(lamb) == str else lamb

    for layer_num in tqdm(range(len(projection_matrices))):
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue

        with torch.no_grad():
            # mat1 = \lambda W + \sum{v k^T}
            mat1 = lamb * projection_matrices[layer_num].weight

            # mat2 = \lambda I + \sum{k k^T}
            mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1],
                                    device=projection_matrices[layer_num].weight.device)

            # aggregate sums for mat1, mat2
            for context, values in zip(contexts, valuess):
                context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                mat1 += for_mat1
                mat2 += for_mat2

            # update projection matrix
            projection_matrices[layer_num].weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2))


def populate_test_set(dataset_fname, begin_idx, end_idx):
    test_set = []
    df = pd.read_csv(dataset_fname)
    for idx in range(begin_idx, end_idx):
        entry = {}
        entry["old"] = df["article"][idx] + " " + df["profession"][idx]
        entry["profession"] = df["profession"][idx]
        edit_to = 'female' if df['stereotype'][idx] == 'male' else 'male'
        entry["new"] = "a " + edit_to + " " + df["profession"][idx]
        entry["prompts"] = []

        for i in range(1, 11):
            entry['prompts'].append(df[f'ex{i}'][idx])

        entry['validation'] = df['validation'][idx]

        test_set.append(entry)

    return test_set
