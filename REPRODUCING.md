# Reproducing Results

## Run an experiment

- Run an experiment, by providing the configuration argument as a json:
  
    ``` 
    python run_experiment --arg "<config_dict>" 
    ```

    - Training progress is available in `/exp/<exp_name>/`. Checkpoints are available in `/exp/<exp_name>/ckpts/`. Training curves and statistics are available in `/exp/<exp_name>/logs/`.
    
    - Logs of the whole process are are saved in `/exp/running.log`.


## Configure an experiment

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

## Hyperparameters
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky" rowspan="3"></th>
    <th class="tg-c3ow" colspan="2">No SQS</th>
    <th class="tg-c3ow" colspan="2">SQS</th>
    <th class="tg-c3ow" colspan="2">SQS+</th>
    <th class="tg-c3ow" colspan="2">No SQS</th>
    <th class="tg-c3ow" colspan="2">SQS</th>
    <th class="tg-c3ow" colspan="2">SQS+</th>
  </tr>
  <tr>
    <th class="tg-c3ow" colspan="6">ProtoNet</th>
    <th class="tg-c3ow" colspan="6">MatchingNet</th>
  </tr>
  <tr>
    <th class="tg-8bgf"><span style="font-style:italic">η</span></th>
    <th class="tg-8bgf">Adv_iter</th>
    <th class="tg-8bgf">η</th>
    <th class="tg-8bgf">Adv_iter</th>
    <th class="tg-8bgf">η</th>
    <th class="tg-8bgf">Adv_iter</th>
    <th class="tg-8bgf">η</th>
    <th class="tg-8bgf">Adv_iter</th>
    <th class="tg-8bgf">η</th>
    <th class="tg-8bgf">Adv_iter</th>
    <th class="tg-8bgf">η</th>
    <th class="tg-8bgf">Adv_iter</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">CIFAR100</td>
    <td class="tg-c3ow">22.0</td>
    <td class="tg-c3ow">4</td>
    <td class="tg-c3ow">31.0</td>
    <td class="tg-c3ow">3</td>
    <td class="tg-c3ow">22.0</td>
    <td class="tg-c3ow">4</td>
    <td class="tg-c3ow">22.0</td>
    <td class="tg-c3ow">4</td>
    <td class="tg-c3ow">31.0</td>
    <td class="tg-c3ow">3</td>
    <td class="tg-c3ow">32.0</td>
    <td class="tg-c3ow">2</td>
  </tr>
  <tr>
    <td class="tg-0pky">MiniImagenet</td>
    <td class="tg-c3ow">22.0</td>
    <td class="tg-c3ow">4</td>
    <td class="tg-c3ow">31.0</td>
    <td class="tg-c3ow">3</td>
    <td class="tg-c3ow">22.0</td>
    <td class="tg-c3ow">4</td>
    <td class="tg-c3ow">22.0</td>
    <td class="tg-c3ow">4</td>
    <td class="tg-c3ow">41.0</td>
    <td class="tg-c3ow">8</td>
    <td class="tg-c3ow">24.5</td>
    <td class="tg-c3ow">5</td>
  </tr>
  <tr>
    <td class="tg-0pky">TieredImagenet</td>
    <td class="tg-c3ow">22.0</td>
    <td class="tg-c3ow">4</td>
    <td class="tg-c3ow">17.0</td>
    <td class="tg-c3ow">4</td>
    <td class="tg-c3ow">22.0</td>
    <td class="tg-c3ow">4</td>
    <td class="tg-c3ow">22.0</td>
    <td class="tg-c3ow">4</td>
    <td class="tg-c3ow">41.0</td>
    <td class="tg-c3ow">9</td>
    <td class="tg-c3ow">22.0</td>
    <td class="tg-c3ow">4</td>
  </tr>
  <tr>
    <td class="tg-0pky">FEMNIST-FS</td>
    <td class="tg-c3ow">22.0</td>
    <td class="tg-c3ow">4</td>
    <td class="tg-c3ow">16.5</td>
    <td class="tg-c3ow">2</td>
    <td class="tg-c3ow">24.0</td>
    <td class="tg-c3ow">2</td>
    <td class="tg-c3ow">22.0</td>
    <td class="tg-c3ow">4</td>
    <td class="tg-c3ow">25.8</td>
    <td class="tg-c3ow">5</td>
    <td class="tg-c3ow">30.0</td>
    <td class="tg-c3ow">8</td>
  </tr>
</tbody>
</table>

## Results
Here are the results present in the paper on different datasets.

### Cifar 100
* **No SQS**

    |                      | 5-support 8-query |
    |----------------------|-------------------|
    | ProtoNet             | 48.07 ± 0.44      |
    | Ind_OT + ProtoNet    | 48.62 ± 0.44      |
    | AQP + ProtoNet       | 48.70 ± 0.42      |
    | ASP + ProtoNet       | 49.54 ± 0.44      |
    | MatchingNet          | 46.03 ± 0.42      |
    | Ind_OT + MatchingNet | 45.77 ± 0.42      |
    | AQP + MatchingNet    | 46.53 ± 0.43      |

* **SQS**

    |                      | 1-support 8-query | 1-support 16-query | 5-support 8-query | 5-support 16-query |
    |----------------------|-------------------|--------------------|-------------------|--------------------|
    | ProtoNet             |                   |                    | 43.15 ± 0.48    |                    |
    | Ind_OT + ProtoNet    | 30.19 ± 0.37     | 30.44 ± 0.33      | 43.62 ± 0.49    | 42.19 ± 0.39      |
    | AQP + ProtoNet       | 31.68 ± 0.39     | 31.34 ± 0.34      | 45.09 ± 0.46   | 44.88 ± 0.39      |
    | ASP + ProtoNet       |                   |                    | 44.55 ± 0.46     |                    |
    | MatchingNet          |                   |                    | 39.89 ± 0.44     |                    |
    | Ind_OT + MatchingNet |                   |                    | 40.82 ± 0.45     |                    |
    | AQP + MatchingNet    |                   |                    | 42.40 ± 0.46     |                    |


* **SQS+**

    |                      | 1-support 8-query | 1-support 16-query | 5-support 8-query | 5-support 16-query |
    |----------------------|-------------------|--------------------|-------------------|--------------------|
    | ProtoNet             |                   |                    | 40.59 ± 0.69    |                    |
    | Ind_OT + ProtoNet    | 29.02 ± 0.36     | 28.87 ± 0.31      | 41.74 ± 0.65    | 37.87 ± 0.40      |
    | AQP + ProtoNet       | 31.66 ± 0.39     | 31.45 ± 0.33      | 45.06 ± 0.46   | 43.73 ± 0.39      |
    | ASP + ProtoNet       |                   |                    | 44.24 ± 0.45     |                    |
    | MatchingNet          |                   |                    | 36.63 ± 0.45     |                    |
    | Ind_OT + MatchingNet |                   |                    | 37.13 ± 0.47     |                    |
    | AQP + MatchingNet    |                   |                    | 41.26 ± 0.46     |                    |



### miniImagenet
* **No SQS**

    |                      | 5-support 8-query |
    |----------------------|-------------------|
    | ProtoNet             | 64.56 ± 0.42      |
    | Ind_OT + ProtoNet    | 63.74 ± 0.42      |
    | AQP + ProtoNet       | 66.81 ± 0.42      |
    | ASP + ProtoNet       | 66.90 ± 0.41      |
    | MatchingNet          | 59.68 ± 0.43      |
    | Ind_OT + MatchingNet | 59.64 ± 0.44      |
    | AQP + MatchingNet    | 62.29 ± 0.42      |

* **SQS**
    |                      | 1-support 8-query | 1-support 16-query | 5-support 8-query | 5-support 16-query |
    |----------------------|-------------------|--------------------|-------------------|--------------------|
    | ProtoNet             |                   |                    | 41.68 ± 0.76    |                    |
    | Ind_OT + ProtoNet    | 30.37 ± 0.43     | 31.94 ± 0.42      | 39.84 ± 0.78    | 39.50 ± 0.50      |
    | AQP + ProtoNet       | 30.59 ± 0.43     | 32.05 ± 0.42      | 42.65 ± 0.57   | 40.42 ± 0.59      |
    | ASP + ProtoNet       |                   |                    | 39.31 ± 0.61     |                    |
    | MatchingNet          |                   |                    | 39.66 ± 0.54     |                    |
    | Ind_OT + MatchingNet |                   |                    | 38.25 ± 0.54     |                    |
    | AQP + MatchingNet    |                   |                    | 42.32 ± 0.52     |                    |

* **SQS+**

    |                      | 1-support 8-query | 1-support 16-query | 5-support 8-query | 5-support 16-query |
    |----------------------|-------------------|--------------------|-------------------|--------------------|
    | ProtoNet             |                   |                    | 35.17 ± 0.78    |                    |
    | Ind_OT + ProtoNet    | 30.23 ± 0.43     | 28.75 ± 0.38      | 34.75 ± 0.80    | 34.94 ± 0.47      |
    | AQP + ProtoNet       | 29.74 ± 0.42     | 31.00 ± 0.43      | 40.61 ± 0.60     | 40.42 ± 0.59      |
    | ASP + ProtoNet       |                   |                    | 40.79 ± 0.60     |                    |
    | MatchingNet          |                   |                    | 35.40 ± 0.52     |                    |
    | Ind_OT + MatchingNet |                   |                    | 33.22 ± 0.50     |                    |
    | AQP + MatchingNet    |                   |                    | 37.90 ± 0.53     |                    |

### tieredImagenet
* **No SQS**

    |                      | 5-support 8-query |
    |----------------------|-------------------|
    | ProtoNet             | 71.04 ± 0.45      |
    | Ind_OT + ProtoNet    | 69.56 ± 0.46      |
    | AQP + ProtoNet       | 69.62 ± 0.45      |
    | ASP + ProtoNet       | 68.87 ± 0.46      |
    | MatchingNet          | 67.85 ± 0.46      |
    | Ind_OT + MatchingNet | 67.79 ± 0.46      |
    | AQP + MatchingNet    | 68.40 ± 0.45      |

* **SQS**

    |                      | 5-support 8-query |
    |----------------------|-------------------|
    | ProtoNet             | 41.59 ± 0.57    |
    | Ind_OT + ProtoNet    | 40.08 ± 0.56    |
    | AQP + ProtoNet       | 45.34 ± 0.60   |
    | ASP + ProtoNet       | 44.47 ± 0.58     |
    | MatchingNet          | 43.30 ± 0.56     |
    | Ind_OT + MatchingNet | 44.27 ± 0.56     |
    | AQP + MatchingNet    | 45.26 ± 0.56     |

* **SQS+**

    |                      | 5-support 8-query |
    |----------------------|-------------------|
    | ProtoNet             | 38.57 ± 0.65     |
    | Ind_OT + ProtoNet    | 35.81 ± 0.58     |
    | AQP + ProtoNet       | 40.94 ± 0.66     |
    | ASP + ProtoNet       | 40.03 ± 0.69     |
    | MatchingNet          | 37.57 ± 0.57     |
    | Ind_OT + MatchingNet | 39.24 ± 0.59     |
    | AQP + MatchingNet    | 39.39 ± 0.58     |


### FEMNIST
* **No **SQS****

    |                      | 1-support 1-query |
    |----------------------|-------------------|
    | ProtoNet             | 93.09 ± 0.51      |
    | Ind_OT + ProtoNet    | 91.66 ± 0.55      |
    | AQP + ProtoNet       | 94.61 ± 0.45      |
    | ASP + ProtoNet       | 91.93 ± 0.48      |
    | MatchingNet          | 93.69 ± 0.49      |
    | Ind_OT + MatchingNet | 93.76 ± 0.48      |
    | AQP + MatchingNet    | 93.69 ± 0.49     |
    

* **SQS**

    |                      | 1-support 1-query |
    |----------------------|-------------------|
    | ProtoNet             | 84.36 ± 0.74     |
    | Ind_OT + ProtoNet    | 79.64 ± 0.80     |
    | AQP + ProtoNet       | 85.92 ± 0.69     |
    | ASP + ProtoNet       | 84.72 ± 0.73     |
    | MatchingNet          | 85.88 ± 0.69     |
    | Ind_OT + MatchingNet | 84.08 ± 0.71     |
    | AQP + MatchingNet    | 87.24 ± 0.67     |

* **SQS+**

    |                      | 1-support 1-query |
    |----------------------|-------------------|
    | ProtoNet             | 82.67 ± 0.77     |
    | Ind_OT + ProtoNet    | 76.37 ± 0.84     |
    | AQP + ProtoNet       | 84.42 ± 0.74     |
    | ASP + ProtoNet       | 84.14 ± 0.73     |
    | MatchingNet          | 83.48 ± 0.74     |
    | Ind_OT + MatchingNet | 83.09 ± 0.74     |
    | AQP + MatchingNet    | 84.98 ± 0.72     |
    
