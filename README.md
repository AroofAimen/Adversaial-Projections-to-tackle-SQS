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

### Run an experiment

- Run an experiment, by providing the configuration argument as a json:
  
    ``` 
    python run_experiment --arg "<config_dict>" 
    ```

    - Training progress is available in `/exp/<exp_name>/`. Checkpoints are available in `/exp/<exp_name>/ckpts/`. Training curves and statistics are available in `/exp/<exp_name>/logs/`.
    
    - Logs of the whole process are are saved in `/exp/running.log`.


### Configure an experiment

Default experiment configuration is present in `/src/configs.py`. One can refer to `configs.py` to know about all the configuration variables.

1. To setup **No SQS** in an experiment, set the following flags,
    ```
    "no_change_perturb_train_s":1,"no_change_perturb_train_q":1,"no_change_perturb_test_s":1,"no_change_perturb_test_q":1,"train_support_query_shift":0,"test_support_query_shift":0
    ```
2. To setup **SQS** in an experiment, set the following flags,
    ```
    "no_change_perturb_train_s":0,"no_change_perturb_train_q":0,"no_change_perturb_test_s":0,"no_change_perturb_test_q":0,"train_support_query_shift":1,"test_support_query_shift":1
    ```
3. To setup **SQS+** in an experiment, set the following flags,
    ```
    "no_change_perturb_train_s":1,"no_change_perturb_train_q":1,"no_change_perturb_test_s":0,"no_change_perturb_test_q":0,"train_support_query_shift":0,"test_support_query_shift":1
    ```

Following is a command to run ProtoNet+AQP on miniImagenet dataset for SQS+ variant.
```
python run_experiment.py --arg '{"exp":"EXP_0","image_size":84,"filter_size":64,"model_name":"ProtoNet","backbone":"resnet18","base_lr":0.1,"meta_lr":0.001,"dataset":"mini-imagenet","train_spec_file":"train.json","val_spec_file":"val.json","test_spec_file":"test.json","n_way":5,"n_source":5,"n_target":8,"n_episodes":400,"n_val_tasks":300,"n_epochs":150,"n_way_eval":5,"n_source_eval":5,"n_target_eval":8,"n_tasks_eval":2000,"no_change_perturb_train_s":1,"no_change_perturb_train_q" :1,"no_change_perturb_test_s":0,"no_change_perturb_test_q":0,"train_support_query_shift":0,"test_support_query_shift":1,"use_train_query":1,"use_test_query":0,"gpu":0,"report_ray":0,"use_adv_project":1,"adv_max_iter":4,"adv_lr":22.0,"train_project_task":1,"val_freq":1,"adv_proj_prob":0.25,"adv_rand_conv_prob":2.0}'

```

* To run experiment without **OT** and **AQP** set `use_adv_project` to 0 and `use_transport_mod` to 0.
* To run experiment with any one of **OT** or **AQP** set corresponding flag `use_adv_project` to 1 or `use_transport_mod` to 1.


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

    Each dataset has `train.json`, `test.json` and `validation.json` files. These files contain the specifications of the domains for train, test, validation phases.

* **FEMNIST**

    FEMNIST's specifications include three files, `train.csv`, `test.csv` and `val.csv`. Each file contains the images belonging to the corresponding phase.
    *Note:* If root location of FEMNIST dataset is changed, then modify the updated location for the images in the .csv files.

## Contact
To ask questions or report issues, please open an issue [here](https://github.com/Few-Shot-SQS/aqp/issues)

## References
FewShiftBed code is modified from https://github.com/ebennequin/meta-domain-shift

