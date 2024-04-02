# KandiSuperRes - diffusion model for 4K super resolution

[Habr Post](https://habr.com/ru/companies/sberbank/articles/775590/) | [![Hugging Face Spaces](https://img.shields.io/badge/🤗-Huggingface-yello.svg)](https://huggingface.co/ai-forever/KandiSuperRes/) | [Telegram-bot](https://t.me/kandinsky21_bot) | [Technical Report](https://arxiv.org/pdf/2312.03511.pdf)| [Our text-to-image model](https://github.com/ai-forever/Kandinsky-3/tree/main)

![](assets/title.png)

## Installing

To install repo first one need to create conda environment:

```
conda create -n kandisuperres -y python=3.8;
source activate kandisuperres;
pip install -r requirements.txt;
```

## How to use:

Check our jupyter notebook `KandiSuperRes.ipynb` with example. 

```python
from KandiSuperRes import get_SR_pipeline

sr_pipe = get_SR_pipeline(device='cuda', fp16=True)

lr_image = ... # PIL Image
sr_image = sr_pipe(lr_image)
```