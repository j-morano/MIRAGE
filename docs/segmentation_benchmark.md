# MIRAGE - Segmentation benckmark


## Download

Compressed datasets with data splits are available in the [Segmentation datasets release](https://github.com/j-morano/MIRAGE/releases/tag/seg-data) on GitHub.
The links below are direct download from the release page.


| Dataset | Download link |
| --- | --- |
| AROI | [Full dataset](https://github.com/j-morano/MIRAGE/releases/download/seg-data/AROI.zip) |
| Duke DME | [Full dataset](https://github.com/j-morano/MIRAGE/releases/download/seg-data/Duke_DME.zip) |
| Duke iAMD | [Part A](https://github.com/j-morano/MIRAGE/releases/download/seg-data/Duke_iAMD_labeled_part_aa) - [Part B](https://github.com/j-morano/MIRAGE/releases/download/seg-data/Duke_iAMD_labeled_part_ab) - [Part C](https://github.com/j-morano/MIRAGE/releases/download/seg-data/Duke_iAMD_labeled_part_ac) - [Part D](https://github.com/j-morano/MIRAGE/releases/download/seg-data/Duke_iAMD_labeled_part_ad) - [Part E](https://github.com/j-morano/MIRAGE/releases/download/seg-data/Duke_iAMD_labeled_part_ae) - [Part F](https://github.com/j-morano/MIRAGE/releases/download/seg-data/Duke_iAMD_labeled_part_af) - [Part G](https://github.com/j-morano/MIRAGE/releases/download/seg-data/Duke_iAMD_labeled_part_ag) - [Part H](https://github.com/j-morano/MIRAGE/releases/download/seg-data/Duke_iAMD_labeled_part_ah) |
| GOALS | [Full dataset](https://github.com/j-morano/MIRAGE/releases/download/seg-data/GOALS.zip) |
| RETOUCH | [Part A](https://github.com/j-morano/MIRAGE/releases/download/seg-data/RETOUCH_part_aa) - [Part B](https://github.com/j-morano/MIRAGE/releases/download/seg-data/RETOUCH_part_ab) |


> [!IMPORTANT]
> Some datasets are split into multiple parts due to size limitations. Check the [Large datasets](#large-datasets) section for more information.


## Data structure

### Training and evaluation dataset

AROI, Duke DME, GOALS, and RETOUCH datasets.

`bscan/` contains the B-scans and `semseg/` contains the segmentation masks, both in PNG format and with the same names.
`INFO.json` contains mapping of the classes to the intensity values in the segmentation maps and the full class names.


```text
Dataset/
    |-- test/
    |    |-- bscan/
    |    |    |-- img_id_0.png
    |    |    | ...
    |    |-- semseg/
    |         |-- img_id_0.png
    |         | ...
    |-- train/
    |    |-- bscan/
    |    |    |-- img_id_1.png
    |    |    | ...
    |    |-- semseg/
    |         |-- img_id_1.png
    |         | ...
    |-- val/
    |    |-- bscan/
    |    |    |-- img_id_2.png
    |    |    | ...
    |    |-- semseg/
    |         |-- img_id_2.png
    |         | ...
    |-- INFO.json
```

Example `INFO.json` (from GOALS):

```json
{
    "0": {
        "value": 0,
        "label": "Elsewhere"
    },
    "1": {
        "value": 80,
        "label": "RNFL"
    },
    "2": {
        "value": 160,
        "label": "GCIPL"
    },
    "3": {
        "value": 255,
        "label": "Choroid"
    }
}
```

> [!IMPORTANT]
> RETOUCH dataset does not have a `semseg` folder under `test` because the test segmentation masks are not available. The evaluation was performed on the Grand Challenge platform (<https://retouch.grand-challenge.org/>).


### Evaluation-only dataset

Duke iAMD (labeled) dataset.

```text
Dataset/
    |-- bscan/
    |    |-- img_id_0.png
    |    |-- img_id_1.png
    |    | ...
    |-- semseg/
    |    |-- img_id_0.png
    |    |-- img_id_1.png
    |    | ...
    |-- INFO.json
```


## Large datasets

Large datasets (Duke iAMD and RETOUCH) are split into multiple parts using the Linux `split` command.

```shell
split -b $(( $(stat -c%s Duke_iAMD_labeled.zip) / 8 + 1 )) Duke_iAMD_labeled.zip Duke_iAMD_labeled_part_
split -b $(( $(stat -c%s RETOUCH.zip) / 2 + 1 )) RETOUCH.zip RETOUCH_part_
```

To reassemble the parts, run:

```shell
cat Duke_iAMD_labeled_part_* > Duke_iAMD_labeled.zip
cat RETOUCH_part_* > RETOUCH.zip
```

In windows, use the `copy` command in the Command Prompt (cmd):

```shell
copy /b Duke_iAMD_labeled_part_* Duke_iAMD_labeled.zip
copy /b RETOUCH_part_* RETOUCH.zip
```



## Official links

| Dataset | Official website |
| --- | --- |
| AROI | [Link](https://ipg.fer.hr/ipg/resources/oct_image_database) |
| Duke DME | [Link](https://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm) |
| Duke iAMD | [Link](https://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm) |
| GOALS | _No longer available_ |
| RETOUCH | [Link](https://retouch.grand-challenge.org/) |

