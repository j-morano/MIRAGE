# Classification tuning

This repository provides the code to tune MIRAGE and other state-of-the-art foundation models for OCT and SLO classification tasks.


## Requirements

See [README#Requirements](../README.md#requirements) for the requirements.


## Data

The OCT segmentation datasets are available in [docs/classification_benchmark.md](../docs/classification_benchmark.md).


## Usage

The script `run_cls_tuning.py` provides the main entry point to tune the models. It supports several command-line arguments to configure the training process.

We also provide the utility script `runner` (see [docs/runner.md](../docs/runner.md)) to run multiple experiments easily by specifying multiple entries for the same argument.
Below we provide an example to tune MIRAGE (both Base and Large) on the Duke DME dataset.


```bash
./runner python run_cls_tuning.py \
    --runners 1 \
    -- \
    --version v1 \
    --seed 0 \
    --weights \
        ./__weights/MIRAGE-Base.pth \
        ./__weights/MIRAGE-Large.pth \
    --linear_probing \
    --data_root \
        ./__datasets/Classification \
    --data_set \
        GAMMA
```

> [!TIP]
> Run the script with the `-h` or `--help` flag to see the available options.


By default, the script will save the model weights and the training logs in the `./__output/cls` directory.
You can specify a different output directory using the `--base_output_dir` argument.

> [!IMPORTANT]
> The script uses the filename of the weights to determine which model configuration to use. In particular, the filename should contain the model name, so that the following substrings load the corresponding model configuration (case-insensitive):
>
> - `mirage-base`: MIRAGE-Base
> - `mirage-large`: MIRAGE-Large



## Adding a new dataset

To add a new dataset, you need to respect the dataset structure indicated in [docs/classification_benchmark.md](../docs/classification_benchmark.md).


## Adding a new model

To add a new model, you need to create a new model class in `fm_cls_config.py` extending the `FoundModel` class in the same file.
You can check the existing adapted models in the file for reference.

