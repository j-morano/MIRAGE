# Pretraining

This repository provides the code to pretrain MIRAGE on a multimodal dataset of OCT, SLO, and layer segmentation images.


## Requirements

See [README#Requirements](../README.md#requirements) for the requirements.


## Data

For privacy reasons, we cannot provide the pretraining dataset used in the paper. In this section, we describe the expected structure of the dataset to use the provided pretraining code.

The code expects the dataset to be organized as follows:

```text
Dataset/
    |-- bscan/
    |    |-- img_id_0.npy
    |    |-- img_id_1.npy
    |    | ...
    |-- slo/
    |    |-- img_id_0.npy
    |    |-- img_id_1.npy
    |    | ...
    |-- bscanlayermap/
         |-- img_id_0.npy
         |-- img_id_1.npy
         | ...
```

All images are expected to be NumPy arrays with the shape `(1, 512, 512)` and the data type `np.uint8` (range [0, 255]).


## Usage

The script `run_pretraining.py` provides the main entry point to pretrain the model. It supports several command-line arguments to configure the training process.
This arguments can also be provided in a configuration file. See [_cfgs/pre_mirage_98_1600e_bscan-slo-bscanlayermap_512-128--32-8.yaml](../_cfgs/pre_mirage_98_1600e_bscan-slo-bscanlayermap_512-128--32-8.yaml) for an example with the default configuration used in the paper.

We also provide the utility script `runner` (see [docs/runner.md](../docs/runner.md)) to run multiple experiments easily by specifying multiple entries for the same argument.
Below we provide an example to pretraing MIRAGE (both Base and Large) on the multimodal dataset.


```bash
./runner python run_pretraining.py \
    --runners 1 \
    -- \
    --config ./_cfgs/pre_mirage_98_1600e_bscan-slo-bscanlayermap_512-128--32-8.yaml \
    --data_path \
        ./__datasets/Pretraining/ \
    --weights \
        ./__weights/multimae-b_98_rgb+-depth-semseg_1600e_multivit-afff3f8c.pth \
        ./__weights/mae_pretrain_vit_large.pth
```

> [!TIP]
> Run the script with the `-h` or `--help` flag to see the available options.


> [!IMPORTANT]
> The script uses the filename of the weights to determine which model configuration to use. In particular, the filename should contain the model name, so that the following substrings load the corresponding model configuration (case-insensitive):
>
> - `multimae-b` (based on MultiMAE): MIRAGE-Base
> - `mae_pretrain` (based on MAE): MIRAGE-Large
>
> Please refer to the [MultiMAE](https://github.com/EPFL-VILAB/MultiMAE) and [MAE](https://github.com/facebookresearch/mae) repositories to download the weights.

