"""
Run a complete experiment (training + evaluation)
"""

from functools import partial
import os
import torch
import pprint

from src.running_steps import Trainer, set_and_print_random_seed
from src.modules import get_backbone, OptimalTransport
from src.methods import get_model
from src.parser import parse_args
from src.adv_train import Adv_lr_schedular

if __name__ == "__main__":

    config = parse_args()
    config.backbone = get_backbone(config)
    if config.use_transport_mod:
        TRANSPORTATION_MODULE = OptimalTransport(
                                    regularization=config.regularization,
                                    learn_regularization=config.learn_regularization,
                                    max_iter=config.transport_max_iter,
                                    stopping_criterion=config.stopping_criterion,
                                    **{"opt_transport_prob":config.opt_transport_prob}
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

    with open(os.path.join(config.LOG_DIR, "config.txt"), "w") as f:
        pprint.pprint(vars(config), f)

    trainer = Trainer(config)

    if config.mode == "train":
        set_and_print_random_seed(config.seed)
        trained_model = trainer.train_model()
        torch.cuda.empty_cache()

        set_and_print_random_seed(config.seed)
        trainer.eval_model(trained_model)
        trainer.save()

    if config.mode == "test":
        set_and_print_random_seed(config.seed)
        trainer.eval_model(trained_model)
        trainer.save()