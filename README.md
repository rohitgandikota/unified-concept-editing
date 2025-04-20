# Unified Concept Editing in Diffusion Models
###  [Project Website](https://unified.baulab.info) | [Arxiv Preprint](https://arxiv.org/pdf/2308.14761.pdf)

## Update: FLUX and SDXL Support for UCE🚀🚀🚀
You can now edit FLUX and SDXL models using UCE under 1 second! Right now it is experimental! We are continuing the developement for FLUX. Expect it to get much better over the next few days. Feel free to share your experience<br>

<div align='center'>
<img src = 'images/intro.png'>
</div>

Model-editing methods can be used to address individual issues of bias, copyright, and offensive content in text-to-image models, but in the real world, all of these issues will appear simultaneously in the same model. We present an algorithm that scales seamlessly to concurrent edits on text-conditional diffusion models. Our method, Unified Concept Editing (UCE) enables debiasing multiple attributes simultaneously while also erasing artistic styles en masse to address copyright and reducing potentially offensive content. <br>

Specifically, we demonstrate scalable simultaneous debiasing, style erasure, and content moderation by editing text-to-image projections. We concurrently debiase multiple professions across gender and race. To address copyright, we erase styles at scale with minimal interference. For content safety, we regulate many unsafe concepts together. Our interpretable editing allows addressing all these issues concurrently, and we present extensive experiments demonstrating improved scalability over prior work.
<div align='center'>
<img src = 'images/method.png'>
</div>

## Installation Guide

The code base is based on the `diffusers` package. To get started:
```
git clone https://github.com/rohitgandikota/unified-concept-editing.git
cd unified-concept-editing
mkdir models
pip install -r requirements.txt
```

## Training Guide

After installation, follow these instructions to train a custom UCE model:
### SDv1.4 and SDXL
To erase concepts (e.g. "Van Gogh" and "Picasso" simultaneously) for SDv1.4
```python
python train-scripts/uce_sd_erase.py --model_id 'CompVis/stable-diffusion-v1-4' --erase_concepts 'Van Gogh; Picasso' --guided_concept 'art' --preserve_concepts 'Monet; Rembrandt; Warhol' --device 'cuda:0' --concept_type 'art' --exp_name 'vangogh_uce_sd'
```
To erase concepts (e.g. "Van Gogh" and "Picasso" simultaneously) for SDXL
```python
python train-scripts/uce_sd_erase.py --model_id 'stabilityai/stable-diffusion-xl-base-1.0' --erase_concepts 'Van Gogh, Picasso' --guided_concept 'art' --preserve_concepts 'Monet; Rembrandt; Warhol' --device 'cuda:0' --concept_type 'art' --exp_name 'vangogh_uce_sdxl'
```
### FLUX
To erase concepts (e.g. "Van Gogh") for FLUX
```python
python train-scripts/uce_flux_erase.py --model_id 'black-forest-labs/FLUX.1-schnell' --erase_concepts 'Van Gogh' --preserve_concepts 'Monet; Rembrandt; Warhol' --device 'cuda:0' --concept_type 'art' --exp_name 'vangogh_uce_flux'
```

### Moderating
### SDv1.4 and SDXL
To moderate concepts (e.g. "violence, nudity, harm")
```python
python train-scripts/uce_sd_erase.py --model_id 'CompVis/stable-diffusion-v1-4' --erase_concepts 'violence; nudity; harm' --device 'cuda:0' --concept_type 'unsafe' --exp_name 'i2p'
```

### FLUX
To moderate concepts (e.g. "violence, nudity, harm")
```python
python train-scripts/uce_flux_erase.py --model_id 'black-forest-labs/FLUX.1-schnell' --erase_concepts 'violence; nudity; harm' --device 'cuda:0' --concept_type 'unsafe' --exp_name 'i2p_flux'
```

### Debiasing
To debias concepts (e.g. "Doctor, Nurse, Carpenter") against attributes (e.g. "Male, Female") 
```python
python train-scripts/train_debias.py --concept 'Doctor; Nurse; Carpenter' --attributes 'male; female' --device 'cuda:0' '
```

## Generation Images
To use `eval-scripts/generate-images.py` you would need a CSV file with columns `prompt`, `evaluation_seed`, and `case_number`. (Sample data in `data/`)
```python
python eval-scripts/generate-images.py --model_id 'CompVis/stable-diffusion-v1-4' --uce_model_path 'uce_models/vangogh.safetensors' --prompts_path 'data/vangogh_prompts.csv' --save_path 'uce_results' --exp_name 'vangogh_uce' --num_images_per_prompt 1 --num_inference_steps 50 --device 'cuda:0'
```

## Citing our work
The preprint can be cited as follows
```
@article{gandikota2023unified,
  title={Unified Concept Editing in Diffusion Models},
  author={Rohit Gandikota and Hadas Orgad and Yonatan Belinkov and Joanna Materzy\'nska and David Bau},
  journal={arXiv preprint arXiv:2308.14761},
  year={2023}
}
```
