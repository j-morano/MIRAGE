# Segmentation tuning

This repository provides the code to tune MIRAGE and other state-of-the-art foundation models for OCT segmentation tasks.


## Requirements

See [README#Requirements](../README.md#requirements) for the requirements.


## Data

The OCT segmentation datasets are available in [docs/segmentation_benchmark.md](../docs/segmentation_benchmark.md).


## Usage

The script `run_seg_tuning.py` provides the main entry point to tune the models. It supports several command-line arguments to configure the training process. In addition, configuration files (in YAML) can be used to specify theses arguments. The default configuration, used for the experiments in the paper, is available in [_cfgs/ft_semseg_200e_convnext.yaml](../_cfgs/ft_semseg_200e_convnext.yaml).

We also provide the utility script `runner` to run multiple experiments easily by specifying multiple entries for the same argument.
Below we provide an example to tune MIRAGE (both Base and Large) on the Duke DME dataset.

```bash
./runner python run_seg_tuning.py \
    --runners 1 \
    -- \
    --version v1 \
    --config \
        ./_cfgs/ft_semseg_200e_convnext.yaml \
    --weights \
        ./__weights/MIRAGE-Base.pth \
        ./__weights/MIRAGE-Large.pth \
    --data_path \
        ./__datasets/Segmentation/Duke_DME/
```

You can run `python run_seg_tuning.py --help` to see the available arguments and their descriptions.
These arguments can also be specified in a YAML configuration file.
See [_cfgs/ft_semseg_200e_convnext.yaml](../_cfgs/ft_semseg_200e_convnext.yaml) for an example and the default configuration used in the paper.

By default, the script will save the model weights and the training logs in the `./__output_seg` directory.
You can specify a different output directory using the `--base_output_dir` argument.

> [!IMPORTANT]
> The script uses the filename of the weights to determine which model configuration to use. In particular, the filename should contain the model name, so that the following substrings load the corresponding model configuration (case-insensitive):
>
> - `mirage-base`: MIRAGE-Base
> - `mirage-large`: MIRAGE-Large
> - `dinov2`: DINOv2
> - `retfound`: RETFound
> - `medsam`: MedSAM
>
> DINOv2 is automatically loaded from `torch.hub`, so using `--weights dinov2` is enough to load the model.
> For RETFound and MedSAM weights, please check the corresponding repositories ([RETFound](https://github.com/rmaphoh/RETFound_MAE), [MedSAM](https://github.com/bowang-lab/MedSAM)).


## Adding a new dataset

To add a new dataset, you need to respect the dataset structure indicated in [docs/segmentation_benchmark.md](../docs/segmentation_benchmark.md).


## Adding a new model

To add a new model, you need to create a new model class in `fm_seg_config.py` extending the `FoundModel` class in the same file.
Your model has to be compatible with the input (patchifier+projector) and output adapters (decoders) used in our codebase.
In general, this means that it has to accept and return tensors of shape `(B, T, D)`, where `B` is the batch size, `T` is the sequence length (number of patches + 1 for the global token), and `D` is the feature dimension. For example, MIRAGE-Base uses by default a sequence length of 1025 (`(image_size // patch_size) ** 2 + 1`, with `image_size=1024` and `patch_size=32`), and a feature dimension of 768.
You can check `MIRAGELight` in [mirage/model.py](../mirage/model.py) for more details on the model structure.
