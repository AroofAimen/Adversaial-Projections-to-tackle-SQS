import os
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from functools import partial
import torch
import pprint

from src.running_steps import Trainer, set_and_print_random_seed
from src.modules import get_backbone, OptimalTransport
from src.adv_train import Adv_lr_schedular
from src.methods import get_model
from src.parser import parse_args

def run_exp(ray_config, checkpoint_dir=None, config=None):
    # update config from ray params
    config.meta_lr = ray_config["meta_lr"]
    if "ANIL" in ray_config:
        config.base_lr = ray_config["base_lr"]
    
    if "adv_max_iter" in ray_config:
        config.adv_max_iter = ray_config["adv_max_iter"]
        if not config.adv_lr_sched:
            config.adv_lr = ray_config["adv_lr"]
    
    if "transport_max_iter" in ray_config:
        config.transport_max_iter = ray_config["transport_max_iter"]
        config.regularization = ray_config["regularization"]

    if config.adv_lr_sched:
        config.adv_lr_gamma = ray_config["adv_lr_gamma"]
    
    # define exp dirs 
    config.EXP_DIR = ray_config["exp_dir"]
    config.LOG_DIR = os.path.join(config.EXP_DIR, 'logs')
    config.CKPT_DIR =os.path.join(config.EXP_DIR, 'ckpt')

    os.makedirs(config.CKPT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # init model
    if config.use_transport_mod:
        TRANSPORTATION_MODULE = OptimalTransport(
                                    regularization=config.regularization,
                                    learn_regularization=config.learn_regularization,
                                    max_iter=config.transport_max_iter,
                                    stopping_criterion=config.stopping_criterion,
                                )
    else:
        TRANSPORTATION_MODULE = None

    if config.use_adv_project:
        ADVERSARIAL_PROJECT_CONFIG = {
                                        "max_T" : config.adv_max_iter,
                                        "lr" : config.adv_lr,
                                        "rand_conv_prob" : config.adv_rand_conv_prob,
                                        "train_project_task" : config.train_project_task,
                                        "test_project_task" : config.test_project_task,
                                        "adv_lr_sched" : Adv_lr_schedular(
                                                            config.adv_lr_gamma, 
                                                            config.adv_lr_max
                                                         ) if config.adv_lr_sched else None
                                    }
    else:
        ADVERSARIAL_PROJECT_CONFIG = None

    config.model = partial(
                    get_model(config.model_name),
                    transportation=TRANSPORTATION_MODULE,
                    adv_train_config=ADVERSARIAL_PROJECT_CONFIG,
                    use_test_query=config.use_test_query,
                    use_train_query=config.use_train_query
                )
    
    with open(os.path.join(config.EXP_DIR, "config.txt"), "w") as f:
        pprint.pprint(vars(config), f)
    
    trainer = Trainer(config)
    set_and_print_random_seed(config.seed)
    trained_model = trainer.train_model()        


def main(config, TRAINER, gpus_per_trial=1, num_samples=10, max_epochs=30):
    ray_config = {
        "exp_dir": './',
        # "meta_lr": tune.loguniform(1e-4, 0.5)
        "meta_lr": 0.001
    }
    scheduler = ASHAScheduler(
                    metric="accuracy",
                    mode="max",
                    max_t=max_epochs,
                    grace_period=4,
                    reduction_factor=2    
                )
    if "ANIL" in config.model_name:
        ray_config["base_lr"] = tune.loguniform(1e-4,1e-2)
        

    if config.use_adv_project:
        ray_config["adv_max_iter"] = tune.choice([2,3,4,5,6,7,8,9])
        ray_config["adv_lr"] = tune.loguniform(15, 50)
        
        
    if config.use_transport_mod:
        ray_config["transport_max_iter"] = tune.choice([n for n in range(500,1501,150)])
        ray_config["regularization"] = tune.loguniform(0.01,1.0)

    if config.adv_lr_sched:
        ray_config["adv_lr_gamma"] = tune.loguniform(0.95, 1.05)
    
    reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])
    out = tune.run(
                partial(TRAINER, config=config),
                resources_per_trial={"cpu": 8, "gpu":gpus_per_trial},
                config           =ray_config,
                num_samples      =num_samples,
                local_dir        =config.EXP_DIR,
                scheduler        =scheduler,
                progress_reporter=reporter,
                name             ="HyperTune",
                log_to_file      =True,
                fail_fast        =True
            )    
    
    best_trial = out.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

if __name__ == "__main__":
    config = parse_args()
    config.backbone = get_backbone(config)
    
    main(
        config=config,
        TRAINER=run_exp,
        gpus_per_trial=1.0,
        num_samples=35,
        max_epochs=30
        )