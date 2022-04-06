# Adversarial Query Projection
This repo contains code accompanying the paper, . It includes code for running few-shot meta-learning experiments on datasets with *No SQS*, *SQS*, *SQS+* settings using *AQP*, *ASP*.

## Getting started

* Clone the repo:

    ```
    git clone https://github.com/Few-Shot-SQS/adversarial-query-projection.git
    ```

* `cd adversarial-query-projection` and create a virtual environment:

    ```
    python3 -m venv env
    source env/bin/activate
    ```

* Install dependencies: 
  ```
  pip install -r requirements.txt
  ```


## Usage

Please refer to [REPRODUCING.md](./REPRODUCING.md)


## Datasets

### Dataset Setup
Our experiments use the following datasets:
- Cifar 100
- Mini-Imagenet
- Tiered-Imagenet
- FEMNIST

Following are the instructions to download the datasets.


* **Mini-Imagenet**

    To download mini Imagenet dataset, we'll use To download tiered Imagenet dataset, we'll use [learn2learn](http://learn2learn.net/) library. Execute the following command to download the dataset,

    ```
    import learn2learn as l2l

    l2l.vision.datasets.MiniImagenet(root='~/data', mode='train', download=True)
    l2l.vision.datasets.MiniImagenet(root='~/data', mode='validation', download=True)
    l2l.vision.datasets.MiniImagenet(root='~/data', mode='test', download=True)
    ```

* **Tiered-Imagenet**
    To download tiered Imagenet dataset, we'll use [learn2learn](http://learn2learn.net/) library. Execute the following command to download the dataset,
    ```
    import learn2learn as l2l

    l2l.vision.datasets.TieredImagenet(root='~/data', mode='train', download=True)
    l2l.vision.datasets.TieredImagenet(root='~/data', mode='validation', download=True)
    l2l.vision.datasets.TieredImagenet(root='~/data', mode='test', download=True)
    ```

* To setup **Cifar 100** and **FEMNIST** datasets follow the instructions present [here](https://github.com/ebennequin/meta-domain-shift/blob/master/DATASETS.md).

### Domain Setup
Cifar 100, Mini-Imagenet and Tiered-Imagenet datasets' domains are built using [Hendrycks' perturbations](https://github.com/hendrycks/robustness). These perturbations are applied online to the images during task creation. FEMNIST dataset contains no perturbations, instead different writers represents different domains.

Dataset specifications (domain, class, image) used in the experiments are present in `./configs/dataset_spec/` folder.

* **Cifar 100, Mini-Imagenet and Tiered-Imagenet**

    Each dataset has `train.json`, `test.json` and `val.json` files. These files contain the specifications of the domains for train, test, validation phases.

* **FEMNIST**

    FEMNIST's specifications include three files, `train.csv`, `test.csv` and `val.csv`. Each file contains the images belonging to the corresponding phase.
    *Note:* If root location of FEMNIST dataset is changed, then modify the updated location for the images in the .csv files.

## Contact
To ask questions or report issues, please open an issue [here](https://github.com/Few-Shot-SQS/aqp/issues)

## References
FewShiftBed code is modified from https://github.com/ebennequin/meta-domain-shift

