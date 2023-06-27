import torch
from diffusers import StableDiffusionPipeline
import numpy as np
import abc
import time_utils
import copy
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--with_to_k', default=False, action='store_true')
parser.add_argument('--do_test', default=False, action='store_true')
parser.add_argument('--edit_func', type=str, choices=['baseline', 'closed_form'])
parser.add_argument('--num_seeds', type=int, help='number of seeds to generate')
parser.add_argument('--begin_idx', type=int)
parser.add_argument('--end_idx', type=int, required=False, default=None)
parser.add_argument('--lamb', type=float, help="lambda for editing")
parser.add_argument('--dataset', type=str, default="../data/gender_bias.csv", required=False)
args, unknown = parser.parse_known_args()
print("SCRIPT ARGUMENTS:")
print(args)
print("---")

with_to_k = args.with_to_k
edit_func = args.edit_func
num_seeds = int(args.num_seeds)
begin_idx = int(args.begin_idx)
end_idx = int(args.end_idx) if args.end_idx is not None else begin_idx+1
dataset = args.dataset
do_test = args.do_test

### load model
LOW_RESOURCE = True
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
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

test_set = time_utils.populate_test_set(dataset, begin_idx, end_idx)

### get projection matrices
ca_clip_layers = [l for l in ca_layers if l.to_v.in_features == 768]
projection_matrices = [l.to_v for l in ca_clip_layers]
og_matrices = [copy.deepcopy(l.to_v) for l in ca_clip_layers]
if with_to_k:
    projection_matrices = projection_matrices + [l.to_k for l in ca_clip_layers]
    og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_clip_layers]

### print number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
params = 0
for l in ca_clip_layers:
    params += l.to_v.in_features * l.to_v.out_features
    if with_to_k:
        params += l.to_k.in_features * l.to_k.out_features
print("Params: ", params)
print("Total params: ", count_parameters(ldm_stable.unet))
print("Percentage: ", (params / count_parameters(ldm_stable.unet)) * 100)

### test set
print("Test set size: ", len(test_set))

old_texts = []
new_texts = []
for curr_item in test_set:
    # print("CURRENT TEST SENTENCE: ", curr_item["old"])

    #### restart LDM parameters
    # num_ca_clip_layers = len(ca_clip_layers)
    # for idx_, l in enumerate(ca_clip_layers):
    #     l.to_v = copy.deepcopy(og_matrices[idx_])
    #     projection_matrices[idx_] = l.to_v
    #     if with_to_k:
    #         l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
    #         projection_matrices[num_ca_clip_layers + idx_] = l.to_k

    #### set up sentences
    old_texts.append(curr_item["old"])
    new_texts.append(curr_item["new"])

print(old_texts, new_texts)

#### prepare input k* and v*
old_embs, new_embs = [], []
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
    old_embs.append(old_emb)
    new_embs.append(new_emb)

idxs_replaces = []
for old_text, new_text in zip(old_texts, new_texts):
    tokens_a = tokenizer(old_text).input_ids
    tokens_b = tokenizer(new_text).input_ids
    tokens_a = [tokenizer.encode("a ")[1] if tokenizer.decode(t) == 'an' else t for t in tokens_a]
    tokens_b = [tokenizer.encode("a ")[1] if tokenizer.decode(t) == 'an' else t for t in tokens_b]
    num_orig_tokens = len(tokens_a)
    num_new_tokens = len(tokens_b)
    idxs_replace = []
    j = 0
    for i in range(num_orig_tokens):
        curr_token = tokens_a[i]
        while tokens_b[j] != curr_token:
            j += 1
        idxs_replace.append(j)
        j += 1
    while j < 77:
        idxs_replace.append(j)
        j += 1
    while len(idxs_replace) < 77:
        idxs_replace.append(76)
    idxs_replaces.append(idxs_replace)

contexts, valuess = [], []
for old_emb, new_emb, idxs_replace in zip(old_embs, new_embs, idxs_replaces):
    context = old_emb.detach()
    values = []
    with torch.no_grad():
        for layer in projection_matrices:
            values.append(layer(new_emb[idxs_replace]).detach())
    contexts.append(context)
    valuess.append(values)

#### define training function
if edit_func == "closed_form":
    print(f"Editing model with lambda {args.lamb}")
    time_utils.edit_closed_form(projection_matrices, contexts, valuess, lamb=args.lamb)


#### set up testing
# saves in "./{s_dir}/{train_f}/{base_prompt}/{category}/{prompt}/seed_{seed}.png"
def run_and_save(prompt, save_dir, profession, seed):
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    images = time_utils.text2image_ldm_stable(ldm_stable, [prompt], latent=None,
                                              num_inference_steps=NUM_DIFFUSION_STEPS,
                                              guidance_scale=GUIDANCE_SCALE, generator=g, low_resource=LOW_RESOURCE)
    os.makedirs(f"./{save_dir}/{profession}/{prompt}/", exist_ok=True)
    full_path = f"./{save_dir}/{profession}/{prompt}/seed_{str(seed)}.png"
    time_utils.view_images(images).save(
        full_path)
    print(f"Saved to {full_path}")

for curr_item in test_set:
    #### run the testing
    kwargs_str = ""
    edit_f_name = edit_func
    if with_to_k: edit_f_name += "_with_to_k"
    if args.lamb is not None: edit_f_name += f"_lamb_{args.lamb}"
    save_dir = f"results/{edit_f_name}"
    profession = curr_item["profession"]
    print("CURRENT TEST: ", profession)
    for seed_ in range(num_seeds):
        run_and_save(curr_item["old"], save_dir, profession, seed_)
        run_and_save(curr_item["validation"], save_dir, profession, seed_)
        if do_test:
            for prompt in curr_item["prompts"]:
                run_and_save(prompt, save_dir, profession, seed_)