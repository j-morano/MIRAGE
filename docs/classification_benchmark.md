# MIRAGE - Classification benckmark


## Download

Compressed datasets with data splits are available in the [Classification datasets release](https://github.com/j-morano/MIRAGE/releases/tag/cls-data) on GitHub.
Kermany dataset is not included in the release, as we use exactly the same data and splits as in the original publication.
However, the link is also provided below.
The links below are direct download from the release page.

| Dataset | Download link |
| --- | --- |
| Duke iAMD | [Link](https://github.com/j-morano/MIRAGE/releases/download/cls-data/Duke_iAMD.zip) |
| GAMMA | [Link](https://github.com/j-morano/MIRAGE/releases/download/cls-data/GAMMA.zip) |
| Harvard Glaucoma | [Link](https://github.com/j-morano/MIRAGE/releases/download/cls-data/Harvard_Glaucoma.zip) |
| Kermany | [Link](https://data.mendeley.com/datasets/rscbjbr9sj/2) |
| Noor Eye Hospital | [Link](https://github.com/j-morano/MIRAGE/releases/download/cls-data/Noor_Eye_Hospital.zip) |
| OCTDL | [Link](https://github.com/j-morano/MIRAGE/releases/download/cls-data/OCTDL.zip) |
| OCTID | [Link](https://github.com/j-morano/MIRAGE/releases/download/cls-data/OCTID.zip) |
| OLIVES | [Link](https://github.com/j-morano/MIRAGE/releases/download/cls-data/OLIVES.zip) |


### Cross-dataset tests

We also provide the dataset splits for the cross-dataset tests.

| Dataset | Download link |
| --- | --- |
| Noor Eye Hospital (cross-dataset test) | [Link](https://github.com/j-morano/MIRAGE/releases/download/cls-data/Noor_Eye_Hospital_cross_test.zip) |
| Noor Eye Hospital (cross-dataset train) | [Link](https://github.com/j-morano/MIRAGE/releases/download/cls-data/Noor_Eye_Hospital_cross_train.zip) |
| UMN + Duke Srinivasan (cross-dataset test) | [Link](https://github.com/j-morano/MIRAGE/releases/download/cls-data/UMN_Duke_Srinivasan_cross_test.zip) |
| UMN + Duke Srinivasan (cross-dataset train) | [Link](https://github.com/j-morano/MIRAGE/releases/download/cls-data/UMN_Duke_Srinivasan_cross_train.zip) |


## Data structure

```text
Dataset/
    |-- test/
    |    |-- Class_0/
    |    |    |-- img_id_0.png
    |    |    | ...
    |    | ...
    |-- train/
    |    |-- Class_0/
    |    |    |-- img_id_1.png
    |    |    | ...
    |    | ...
    |-- val/
         |-- Class_0/
         |    |-- img_id_2.png
         |    | ...
         | ...
```



## Official links

| Dataset | Official website |
| --- | --- |
| Duke iAMD | [Link](https://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm) |
| Duke Srinivasan | [Link](https://people.duke.edu/~sf59/Srinivasan_BOE_2014_dataset.htm) |
| GAMMA | [Link](https://gamma.grand-challenge.org/) |
| Harvard Glaucoma | [Link](https://github.com/Harvard-Ophthalmology-AI-Lab/Harvard-GDP=) |
| Kermany | [Link](https://data.mendeley.com/datasets/rscbjbr9sj/2) |
| Noor Eye Hospital | [Link](https://hrabbani.site123.me/available-datasets/dataset-for-oct-classification-50-normal-48-amd-50-dme) |
| OCTDL | [Link](https://data.mendeley.com/datasets/sncdhf53xc/4) |
| OCTID | [Link](https://borealisdata.ca/dataverse/OCTID) |
| OLIVES | [Link](https://zenodo.org/records/7105232) |
| UMN | [Link](https://people.ece.umn.edu/users/parhi/.DATA/) |
