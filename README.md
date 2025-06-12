<a href=https://arxiv.org/abs/2506.08900><img src="https://img.shields.io/badge/arXiv-2506.08900-red?logo=arXiv&logoColor=white"/></a>

![MIRAGE](https://github.com/user-attachments/assets/c00d2553-a2ad-4bf6-bae0-07d598a1be25)


<p align="center">
    <a href="#quick-start">Quick Start</a> •
    <a href="#model-weights">Weights</a> •
    <a href="#inference">Inference</a> •
    <a href="#evaluation-benchmark">Benchmark</a> •
    <a href="#tuning">Tuning</a> •
    <a href="#citation">Citation</a>
</p>

This repository contains the code for the paper "MIRAGE: A multimodal foundation model and benchmark for comprehensive retinal OCT image analysis".

#### [[`arXiv`](https://arxiv.org/abs/2506.08900)]
<br>


MIRAGE is a multimodal foundation model for comprehensive retinal OCT/SLO image analysis. It is trained on a large-scale dataset of multimodal data, and is designed to perform a wide range of tasks, including disease staging, diagnosis, and layer and lesion segmentation. MIRAGE is based on the [MultiMAE](https://github.com/EPFL-VILAB/MultiMAE) architecture, and is pre-trained using a multi-task learning strategy. The model, based on [ViT](https://github.com/google-research/vision_transformer), is available in two sizes: MIRAGE-Base and MIRAGE-Large. The code in this repository provides the model weights and the code to run inference.



> [!IMPORTANT]
> All scripts and code are intended to run on [Linux](https://github.com/torvalds/linux) systems.


## Overview

![Overview](https://github.com/user-attachments/assets/17548d43-46c0-476c-b006-dbe6b286e82c)


**Overview of the proposed model (MIRAGE) and other general (DINOv2) and domain-specific (MedSAM, RETFound) foundation models.**
In contrast to existing unimodal foundation models, our approach utilizes multimodal self-supervised learning to train a Vision Transformer on a large dataset of paired multimodal retinal images, including optical coherence tomography (OCT), scanning laser ophthalmoscopy (SLO), and automatically generated labels for retinal layers.
We evaluated the model on a comprehensive benchmark consisting of 19 tasks from 14 publicly available datasets and two private datasets, covering both OCT and SLO classification and segmentation tasks. Statistical significance was calculated using the Wilcoxon signed-rank test across all datasets.
Our foundation model, MIRAGE, significantly outperforms state-of-the-art foundation models across all task types.


## TODO

- [x] Basic code to load the model and run inference
- [x] Model weights
- [x] Downstream classification datasets
- [x] Downstream segmentation datasets
- [x] Classification tuning code
- [x] Classification evaluation code
- [x] Segmentation tuning code
- [x] Segmentation evaluation code
- [x] Detailed documentation
- [x] Quick start script
- [x] Pretraining code



## Quick start

For a quick start, use the provided script [prepare_env.py](prepare_env.py) to create a new python environment, install the required packages, and download the model weights and the datasets.

> [!IMPORTANT]
> The script will download the model weights and the datasets, which are large files. Make sure you have enough disk space and a stable internet connection.
>
> In addition, it will install Python 3.10.16 (from source) in the same folder if it detects that the system Python version is not 3.10.*.


```bash
./prepare_env.py
```

> [!TIP]
> Run the script with the `-h` or `--help` flag to see the available options.


## Requirements

> [!NOTE]
> The code has been tested with PyTorch 2.5.1 (CUDA 11.8) and Python 3.10.10.


### pip

Create a new python environment and activate it:
```bash
python -m venv venv  # if not already created
source venv/bin/activate
```

Install the required packages:
```bash
pip install -r requirements.txt
```


## Model weights

The model weights are available in the [Model weights release](https://github.com/j-morano/MIRAGE/releases/tag/weights) on GitHub.

| Model | Link |
| --- | --- |
| MIRAGE-Base | [Weights-Base](https://github.com/j-morano/MIRAGE/releases/download/weights/MIRAGE-Base.pth) |
| MIRAGE-Large | [Weights-Large](https://github.com/j-morano/MIRAGE/releases/download/weights/MIRAGE-Large.pth) |


## Inference

The script `mirage_wrapper.py` provides a simple pipeline to load the model and run inference on a single sample.
This sample is already included in the repository (`_example_images/`) and consists of a triplet of OCT, SLO, and layer segmentation images.

To run the inference, simply execute the script:
```bash
python mirage_wrapper.py
```

Check the code for more details.



## Evaluation benchmark

We provide all the publicly available datasets used in the benchmark with the data splits.
See [docs/segmentation_benchmark.md](docs/segmentation_benchmark.md) for more details on the segmentation benchmark, and [docs/classification_benchmark.md](docs/classification_benchmark.md) for the classification benchmark.


## Pretraining

Although we do not provide the pretraining data due to privacy concerns, we provide the code to pretrain MIRAGE on a multimodal dataset.
Please check the [docs/pretraining.md](docs/pretraining.md) for more details.


## Tuning

We provide the code to fine-tune MIRAGE and other state-of-the-art foundation models for OCT segmentation tasks.
Please check the [docs/segmentation_tuning.md](docs/segmentation_tuning.md) for more details.

We also provide the code to fine-tune the models for OCT and SLO classification tasks.
More information can be found in the [docs/classification_tuning.md](docs/classification_tuning.md) file.


## Questions and issues

If you have any questions or find problems with the code, please open an issue on GitHub.


## Citation

If you find this repository useful, please consider giving it a star ⭐ and a citation 📝:

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

## License

The models and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. See [LICENSE](LICENSE) for more details.



## Acknowledgements

MIRAGE code is mainly based on MultiMAE, along with timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv, MAE, mmsegmentation, MONAI, and RETFound.
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
* <https://github.com/Project-MONAI/MONAI>
* <https://github.com/rmaphoh/RETFound_MAE>
