# MIRAGE

This repository contains the code for the paper "MIRAGE: A multimodal foundation model and benchmark for comprehensive retinal OCT image analysis", submitted to npj Digital Medicine.

MIRAGE is a multimodal foundation model for comprehensive retinal OCT image analysis. It is trained on a large-scale dataset of multimodal data, and is designed to perform a wide range of tasks, including disease staging, diagnosis, and layer and lesion segmentation. MIRAGE is based on the MultiMAE architecture, and is pre-trained using a multi-task learning strategy. The model is available in two sizes: MIRAGE-Base and MIRAGE-Large. The code in this repository provides the model weights and the code to run inference.

> [!IMPORTANT]
> This repository is under construction. The training and evaluation code will be added upon acceptance of the paper.


## TODO

- [x] Basic code to load the model and run inference
- [x] Model weights
- [ ] Training code
- [ ] Evaluation code
- [ ] Downstream datasets
- [ ] Detailed results



## Requirements

The code has been tested with PyTorch 2.5.1 (CUDA 11.8) and Python 3.10.10.


To install the required packages, run:
```bash
pip install -r requirements.txt
```


## Model weights

The model weights are available at Google Drive:

| Model | Link |
| --- | --- |
| MIRAGE-Base | [Weights](https://drive.google.com/file/d/1x0Z8jz6jMOYfDdxFrFSORFEvVz6_Jdns/view?usp=sharing) |
| MIRAGE-Large | [Weights](https://drive.google.com/file/d/1b34P2LixvRknYAqaWVMmZkCkm4XKn4MV/view?usp=sharing) |



## Citation

If you use this code or the model weights, please cite the following paper:

```
@article{morano2025mirage,
  title={MIRAGE: A multimodal foundation model and benchmark for comprehensive retinal OCT image analysis},
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



## Acknowledgements

MIRAGE code is mainly based on MultiMAE code base, along with timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv, and MAE.

* <https://github.com/EPFL-VILAB/MultiMAE>
* <https://github.com/rwightman/pytorch-image-models/tree/master/timm>
* <https://github.com/facebookresearch/deit>
* <https://github.com/facebookresearch/dino>
* <https://github.com/facebookresearch/moco-v3>
* <https://github.com/microsoft/unilm/tree/master/beit>
* <https://github.com/BUPT-PRIV/MAE-priv>
* <https://github.com/facebookresearch/mae>
