# KandiSuperRes - diffusion model for 4K super resolution

[KandiSuperRes Flash Post](https://habr.com/ru/companies/sberbank/articles/805337/) | [KandiSuperRes Post](https://habr.com/ru/companies/sberbank/articles/805337/) | [![Hugging Face Spaces](https://img.shields.io/badge/🤗-Huggingface-yello.svg)](https://huggingface.co/ai-forever/KandiSuperRes/) | [Telegram-bot](https://t.me/kandinsky21_bot) | [Our text-to-image model](https://github.com/ai-forever/Kandinsky-3/tree/main)

## KandiSuperRes Flash

![](assets/title_flash.png)

### Description

KandiSuperRes Flash is a new version of the diffusion model for super resolution. This model includes a distilled version of the KandiSuperRes model and a distilled model [Kandinsky 3.0 Flash](https://github.com/ai-forever/Kandinsky-3/tree/main). KandiSuperRes Flash not only improves image clarity, but also corrects artifacts, draws details, improves image aesthetics. And one of the most important advantages is the ability to use the model in the "infinite super resolution" mode. For more information: details of architecture and training, example of generations check out our [Habr post](https://habr.com/ru/companies/sberbank/articles/805337/).

### Installing

To install repo first one need to create conda environment:

```
conda create -n kandisuperres -y python=3.12;
source activate kandisuperres;
pip install -r requirements.txt;
```

### How to use

Check our jupyter notebook `KandiSuperRes.ipynb` with example. 

```python
from KandiSuperRes import get_SR_pipeline
from PIL import Image

sr_pipe = get_SR_pipeline(device='cuda', fp16=True, flash=True, scale=2)

lr_image = Image.open('')
sr_image = sr_pipe(lr_image)
```

### Infinite super resolution

With KandiSuperRes Flash you can infinitely enlarge images to x16 and more.

![](assets/infinity1.png)
![](assets/infinity2.png)

## KandiSuperRes

![](assets/title.png)

### Description

KandiSuperRes is an open-source diffusion model for x4 super resolution. This model is based on the [Kandinsky 3.0](https://github.com/ai-forever/Kandinsky-3/tree/main) architecture with some modifications. For generation in 4K, the [MultiDiffusion](https://arxiv.org/pdf/2302.08113.pdf) algorithm was used, which allows to generate panoramic images. For more information: details of architecture and training, example of generations check out our [Habr post](https://habr.com/ru/companies/sberbank/articles/805337/).

### How to use

Check our jupyter notebook `KandiSuperRes.ipynb` with example. 

```python
from KandiSuperRes import get_SR_pipeline
from PIL import Image

sr_pipe = get_SR_pipeline(device='cuda', fp16=True, flash=False, scale=4)

lr_image = Image.open('')
sr_image = sr_pipe(lr_image)
```

### Authors
+ Anastasia Maltseva [Github](https://github.com/NastyaMittseva)
+ Vladimir Arkhipkin: [Github](https://github.com/oriBetelgeuse)
+ Andrey Kuznetsov: [Github](https://github.com/kuznetsoffandrey), [Blog](https://t.me/complete_ai)
+ Denis Dimitrov: [Github](https://github.com/denndimitrov), [Blog](https://t.me/dendi_math_ai)