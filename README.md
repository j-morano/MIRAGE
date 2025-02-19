# MIRAGE

This repository contains the code for the paper "MIRAGE: A multimodal foundation model and benchmark for comprehensive retinal OCT image analysis", submitted to _npj Digital Medicine_.

MIRAGE is a multimodal foundation model for comprehensive retinal OCT/SLO image analysis. It is trained on a large-scale dataset of multimodal data, and is designed to perform a wide range of tasks, including disease staging, diagnosis, and layer and lesion segmentation. MIRAGE is based on the [MultiMAE](https://github.com/EPFL-VILAB/MultiMAE) architecture, and is pre-trained using a multi-task learning strategy. The model, based on [ViT](https://github.com/google-research/vision_transformer), is available in two sizes: MIRAGE-Base and MIRAGE-Large. The code in this repository provides the model weights and the code to run inference.

> [!IMPORTANT]
> This repository is under construction. The training and evaluation code will be added upon acceptance of the paper.


## Overview

![Overview](https://github.com/user-attachments/assets/cc9f1d37-3097-4ffb-8e1d-78f176d4e4f0)


**Overview of the proposed model (MIRAGE) and other general (DINOv2) and domain-specific (MedSAM, RETFound) foundation models.**
In contrast to existing unimodal foundation models, our approach utilizes multimodal self-supervised learning to train a Vision Transformer on a large dataset of paired multimodal retinal images, including optical coherence tomography (OCT), scanning laser ophthalmoscopy (SLO), and automatically generated labels for retinal layers.
We evaluated the model on a comprehensive benchmark consisting of 19 tasks from 14 publicly available datasets and two private datasets, covering both OCT and SLO classification and segmentation tasks. Statistical significance was calculated using the Wilcoxon signed-rank test across all datasets.
Our foundation model, MIRAGE, significantly outperforms state-of-the-art foundation models across all task types.


## TODO

- [x] Basic code to load the model and run inference
- [x] Model weights
- [x] Downstream classification datasets
- [x] Downstream segmentation datasets
- [ ] Segmentation evaluation code
- [ ] Detailed segmentation results
- [ ] Classification evaluation code
- [ ] Detailed classification results
- [ ] Pretraining code



## Requirements

> [!NOTE]
> The code has been tested with PyTorch 2.5.1 (CUDA 11.8) and Python 3.10.10.


### pip

Create a new python environment and activate it:
```bash
python -m venv env  # if not already created
source env/bin/activate
```

Install the required packages:
```bash
pip install -r requirements.txt
```


## Model weights

The model weights are available in the [Model weights release](https://github.com/j-morano/MIRAGE/releases/tag/weights) on GitHub.

| Model | Link |
| --- | --- |
| MIRAGE-Base | [Weights](https://github.com/j-morano/MIRAGE/releases/download/weights/MIRAGE-Base.pth) |
| MIRAGE-Large | [Weights](https://github.com/j-morano/MIRAGE/releases/download/weights/MIRAGE-Large.pth) |


## Benchmark

We provide all the publicly available datasets used in the benchmark with the data splits.
See [docs/segmentation_benchmark.md](docs/segmentation_benchmark.md) and [docs/classification_benchmark.md](docs/classification_benchmark.md) for more details on the classification and segmentation benchmarks, respectively.


## Citation

If you find this repository useful, please consider giving it a star ⭐ and a citation 📝:

```
@article{morano2025mirage,
  title={{MIRAGE}: A multimodal foundation model and benchmark for comprehensive retinal {OCT} image analysis},
  author={José Morano
  and Botond Fazekas
  and Emese Sükei
  and Ronald Fecso
  and Taha Emre
  and Markus Gumpinger
  and Georg Faustmann
  and Marzieh Oghbaie
  and Ursula Schmidt-Erfurth
  and Hrvoje Bogunović},
  journal={Preprint},
  year={2025}
}
```

## License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.



## Acknowledgements

MIRAGE code is mainly based on MultiMAE, along with timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv, MAE, and mmsegmentation.
We thank the authors for making their code available.

* <https://github.com/EPFL-VILAB/MultiMAE>
* <https://github.com/rwightman/pytorch-image-models/tree/master/timm>
* <https://github.com/facebookresearch/deit>
* <https://github.com/facebookresearch/dino>
* <https://github.com/facebookresearch/moco-v3>
* <https://github.com/microsoft/unilm/tree/master/beit>
* <https://github.com/BUPT-PRIV/MAE-priv>
* <https://github.com/facebookresearch/mae>
* <https://github.com/open-mmlab/mmsegmentation>
