---
license: cc-by-nc-nd-4.0
library_name: pytorch
pipeline_tag: image-feature-extraction
tags:
- OCT
- SLO
- retinal-imaging
- classification
- segmentation
- pytorch_model_hub_mixin
- model_hub_mixin
---

# MIRAGE-Base

This repo contains the the official weights of the MIRAGE-Base model (based on ViT-Base), from the paper ["MIRAGE: Multimodal foundation model and benchmark for comprehensive retinal OCT image analysis"](http://www.arxiv.org/abs/2506.08900), by José Morano et al. (2025).

Project page: [MIRAGE](https://github.com/j-morano/MIRAGE).

![Overview](https://github.com/user-attachments/assets/17548d43-46c0-476c-b006-dbe6b286e82c)

## MIRAGE models

Model | Resolution | Weights
--- | --- | ---
MIRAGE-Base  | 512x512 | [Download](https://huggingface.co/j-morano/MIRAGE-Base)
MIRAGE-Large | 512x512 | [Download](https://huggingface.co/j-morano/MIRAGE-Large)

## Usage

The model can be loaded using the `PyTorchModelHubMixin` from the `huggingface_hub` package and the code from the `mirage_hf.py` script that can be downloaded from [here](https://raw.githubusercontent.com/j-morano/MIRAGE/refs/heads/main/hf/mirage_hf.py).

```python
from huggingface_hub import PyTorchModelHubMixin
from mirage_hf import MIRAGEWrapper


class MIRAGEhf(MIRAGEWrapper, PyTorchModelHubMixin):
    def __init__(
        self,
        input_size=512,
        patch_size=32,
        modalities='bscan-slo',
        size='base',
    ):
        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            modalities=modalities,
            size=size,
        )

# For the MIRAGE model based on ViT-Base
model = MIRAGEhf.from_pretrained("j-morano/MIRAGE-Base")
# For the MIRAGE model based on ViT-Large
model = MIRAGEhf.from_pretrained("j-morano/MIRAGE-Large")
```

## Citation

If you use our code or our model in your research, we would greatly appreciate it if you give a star to the [repo](https://github.com/j-morano/MIRAGE) and cite our [work](http://www.arxiv.org/abs/2506.08900):

```
@misc{morano2025mirage,
    title={{MIRAGE}: Multimodal foundation model and benchmark for comprehensive retinal {OCT} image analysis},
    author={José Morano and Botond Fazekas and Emese Sükei and Ronald Fecso and Taha Emre and Markus Gumpinger and Georg Faustmann and Marzieh Oghbaie and Ursula Schmidt-Erfurth and Hrvoje Bogunović},
    year={2025},
    eprint={2506.08900},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2506.08900},
}
```
