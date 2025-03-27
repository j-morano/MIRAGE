# Segmentation tuning

This repository provides the code to tune MIRAGE and other state-of-the-art foundation models for OCT segmentation tasks.


## Requirements

See [README#Requirements](../README.md#requirements) for the requirements.


## Data

The OCT segmentation datasets are available in [docs/segmentation_benchmark.md](../docs/segmentation_benchmark.md).


## Usage

The script `run_seg_tuning.py` provides the main entry point to tune the models. It supports several command-line arguments to configure the training process. In addition, configuration files (in YAML) can be used to specify theses arguments. The default configuration, used for the experiments in the paper, is available in [_cfgs/seg_200e_convnext.yaml](../_cfgs/seg_200e_convnext.yaml).

We also provide the utility script `runner` (see [docs/runner.md](../docs/runner.md)) to run multiple experiments easily by specifying multiple entries for the same argument.
Below we provide an example to tune MIRAGE (both Base and Large) on the Duke DME dataset.


```bash
./runner python run_seg_tuning.py \
    --runners 1 \
    -- \
    --version v1 \
    --config \
        ./_cfgs/seg_200e_convnext.yaml \
    --weights \
        ./__weights/MIRAGE-Base.pth \
        ./__weights/MIRAGE-Large.pth \
    --data_path \
        ./__datasets/Segmentation/Duke_DME/
```

> [!TIP]
> Run the script with the `-h` or `--help` flag to see the available options.


By default, the script will save the model weights and the training logs in the `./__output/seg` directory.
You can specify a different output directory using the `--base_output_dir` argument.

> [!IMPORTANT]
> The script uses the filename of the weights to determine which model configuration to use. In particular, the filename should contain the model name, so that the following substrings load the corresponding model configuration (case-insensitive):
>
> - `mirage-base`: MIRAGE-Base
> - `mirage-large`: MIRAGE-Large


## Evaluation

To evaluate the models on the segmentation tasks, first train the models using the script above, and then get the predictions on the test set using the same script with the `--infer_only` and `--test` flags, as shown below.

```bash
./runner python run_seg_tuning.py \
    --runners 1 \
    -- \
    --version v1 \
    --config \
        ./_cfgs/seg_200e_convnext.yaml \
    --weights \
        ./__weights/MIRAGE-Base.pth \
        ./__weights/MIRAGE-Large.pth \
    --infer_only --test \
    --data_path \
        ./__datasets/Segmentation/Duke_DME/
```

Once the predictions are saved, you can evaluate the models using the `run_seg_eval.py` script, which computes volume Dice and IoU as well as 95% Hausdorff distance and saves the results in a CSV file.

```bash
./runner python eval_seg.py \
    --runners 1 \
    -- \
    --model_path \
        ./__output/seg/v1/Duke_DME/MIRAGE-Base_frozen_convnext_CEGDice/ \
        ./__output/seg/v1/Duke_DME/MIRAGE-Large_frozen_convnext_CEGDice/
```

> [!TIP]
> Run the script with the `-h` or `--help` flag to see the available options.


## Adding a new dataset

To add a new dataset, you need to respect the dataset structure indicated in [docs/segmentation_benchmark.md](../docs/segmentation_benchmark.md).


## Adding a new model

To add a new model, you need to create a new model class in `fm_seg_config.py` extending the `FoundModel` class in the same file.
Your model has to be compatible with the input (patchifier+projector) and output adapters (decoders) used in our codebase.
In general, this means that it has to accept and return tensors of shape `(B, T, D)`, where `B` is the batch size, `T` is the sequence length (number of patches + 1 for the global token), and `D` is the feature dimension. For example, MIRAGE-Base uses by default a sequence length of 1025 (`(image_size // patch_size) ** 2 + 1`, with `image_size=1024` and `patch_size=32`), and a feature dimension of 768.
You can check `MIRAGELight` in [mirage/model.py](../mirage/model.py) for more details on the model structure.
