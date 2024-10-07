
<!-- <p align="center"><img src="hpsv2/assets/hps_banner.png"/ width="100%"><br></p> -->
# SePPO: Semi-Policy Preference Optimization for Diffusion Alignment.


This is the official repository for the paper: [SePPO: Semi-Policy Preference Optimization for Diffusion Alignment](). 

# Structure

- `utils/` has the scoring models for evaluation or AI feedback (PickScore, HPS, Aesthetics, CLIP)
- `valid_scores.py` is score results from the pretrained model.
- `requirements.txt` Basic pip requirements.
- `test.py` You can write your prompt in this file to get an image result output by our trained model. 



# Setup

`pip install -r requirements.txt`

# Score Validation

1. Download the pretained SD-1.5 unet checkpoint from [DwanZhang/SePPO](https://huggingface.co/DwanZhang/SePPO) to weight folder


2. Then download the HPSv2 test data from [HPSv2](https://github.com/tgxs002/HPSv2?tab=readme-ov-file). Put it at any where you want.

Then run:

```
python valid_scores.py --token 'your hf token' --unet_checkpoint '/SePPO/weight' --dataset_name 'pickapic'
```
## Important Args

- `--token` your huggingface token
- `--unet_checkpoint` the path to pretrained unet weight
- `--dataset_name` dataset you want to evaluate, can be choosed from ["pickapic_valid", "parti_prompt", 'hpsv2']


# Prompt Testing

Run 

```
python test.py --unet_checkpoint '/SePPO/weight' --prompt 'your prompt'
```
The output image will be saved to main directory.


# Acknowledge

Thanks to [DiffusionDPO](https://github.com/SalesforceAIResearch/DiffusionDPO) and [Diffusers](https://github.com/huggingface/diffusers).

