from collections import OrderedDict
import os
from pathlib import Path
import random
import sys
from ray import tune

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from src.utils import (
                set_device,
                elucidate_ids,
                get_episodic_loader,
                save_history,
                save_ckpt,
                load_ckpt,
                load_history
                )


def set_and_print_random_seed(seed=None):
    """
    Set and print numpy random seed, for reproducibility of the training,
    and set torch seed based on numpy random seed
    Returns:
        int: numpy random seed

    """
    random_seed = seed
    if not random_seed:
        random_seed = np.random.randint(0, 2 ** 32 - 1)
    np.random.seed(random_seed)
    # torch.manual_seed(np.random.randint(0, 2 ** 32 - 1))
    torch.manual_seed(random_seed)
    # random.seed(np.random.randint(0, 2 ** 32 - 1))
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed : {random_seed}")

    return random_seed

class Trainer:
    def __init__(self, config):
        self.config = config
        self.iteration = 0
        
        logger.info("Initializing data loaders...")
        self.train_loader, _ = get_episodic_loader(
            split="train",
            n_way=self.config.n_way,
            n_source=self.config.n_source,
            n_target=self.config.n_target,
            n_episodes=self.config.n_episodes,
            no_change_perturb_s=self.config.no_change_perturb_train_s,
            no_change_perturb_q=self.config.no_change_perturb_train_q,
            support_query_shift=self.config.train_support_query_shift,
            image_size=self.config.image_size,
            dataset=self.config.dataset,
            data_dir=self.config.DATA_DIR,
            spec_file=self.config.TRAIN_SPEC_FILE
        )
        self.val_loader, _ = get_episodic_loader(
            split="val",
            n_way=self.config.n_way,
            n_source=self.config.n_source,
            n_target=self.config.n_target,
            n_episodes=self.config.n_val_tasks,
            no_change_perturb_s=self.config.no_change_perturb_test_s,
            no_change_perturb_q=self.config.no_change_perturb_test_q,
            support_query_shift=self.config.test_support_query_shift,
            image_size=self.config.image_size,
            dataset=self.config.dataset,
            data_dir=self.config.DATA_DIR,
            spec_file=self.config.VAL_SPEC_FILE
        )
        if self.config.test_set_validation_freq:
               self.test_loader, _ = get_episodic_loader(
                split="test",
                n_way=self.config.n_way,
                n_source=self.config.n_source,
                n_target=self.config.n_target,
                n_episodes=self.config.n_val_tasks,
                no_change_perturb_s=self.config.no_change_perturb_test_s,
                no_change_perturb_q=self.config.no_change_perturb_test_q,
                support_query_shift=self.config.test_support_query_shift,
                image_size=self.config.image_size,
                dataset=self.config.dataset,
                data_dir=self.config.DATA_DIR,
                spec_file=self.config.TEST_SPEC_FILE
            )
        
        logger.info("Initializing model...")
        self.model =  set_device(self.config.model(
                                self.config.backbone,
                                **{"n_episodes":self.config.n_episodes,
                                   "model_name":self.config.model_name,
                                   "n_way":self.config.n_way,
                                   "base_lr":self.config.base_lr,
                                   "adv_proj_prob":self.config.adv_proj_prob,
                                   "n_source": self.config.n_source,
                                   "n_target": self.config.n_target,
                                   "plot_path": os.path.join(self.config.LOG_DIR, "Images"),
                                   "plot_perturbed": self.config.plot_perturbed
                                   }
                            ))
        
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                            lr=self.config.meta_lr,
                                            betas=(0.9, 0.999), 
                                            eps=1e-08, 
                                            weight_decay=self.config.weight_decay, 
                                            amsgrad=False)
        self.last_iter = 0
        
        if self.config.resume:
            self.load(self.config.ckpt_no)
        else:
            self.init_history()
        
    def init_history(self):
        self.last_eps  = 0
        self.iteration = 0
        self.history   = {
                        "iterations":{"train":[],"val":[],"test":[],"eval":[]},
                        "loss":{"train":[],"val":[],"test":[],"eval":[]},
                        "accuracy":{"train":[],"val":[],"test":[],"eval":[]},
                        "loss-std":{"train":[],"val":[],"test":[],"eval":[]},
                        "accuracy-std":{"train":[],"val":[],"test":[],"eval":[]},
                        }
    
    def log_history(self, mode, iteration, loss, acc):
        self.history["iterations"][mode].append(iteration)
        self.history["loss"][mode].append(np.mean(loss))
        self.history["accuracy"][mode].append(np.mean(acc))
        self.history["loss-std"][mode].append(np.std(loss))
        self.history["accuracy-std"][mode].append(np.std(acc))
        
    def save(self, _save_ckpt=True):
        save_history(self.history, self.config.model_name, self.config.LOG_DIR)
        if not self.config.report_ray and _save_ckpt:
            save_ckpt(self.history, self.iteration, self.model, self.optimizer, self.config.CKPT_DIR)
    
    def load(self, ckpt_no):
        # self.history = load_history(self.config.LOG_DIR)
        self.model, self.optimizer, self.last_iter, self.history = load_ckpt(
                                            self.model, self.optimizer,
                                            self.config.dev, self.config.CKPT_DIR,
                                            ckpt_no, logger
                                        )
    
    def train_model(self):
        max_acc = -1.0
        best_model_epoch = -1
        best_model_state = None

        writer = SummaryWriter(log_dir=self.config.EXP_DIR)

        logger.info("Model and data are ready. Starting training...")
        for self.iteration in range(self.last_iter+1, self.config.n_epochs+1):
            # Set model to training mode
            self.model.train()
            # Execute a training loop of the model
            train_loss, train_acc = self.model.train_loop(self.iteration, self.train_loader, self.optimizer)
            writer.add_scalar("Train/loss", train_loss.mean(), self.iteration)
            writer.add_scalar("Train/acc", train_acc.mean(), self.iteration)
            
            self.log_history('train', self.iteration, train_loss, train_acc)
            
            if (self.iteration)%self.config.val_freq == 0:
                # Set model to evaluation mode
                self.model.eval()
                # Evaluate on validation set
                val_loss, val_acc, _ = self.model.eval_loop(self.val_loader)
                writer.add_scalar("Val/loss", val_loss.mean(), self.iteration)
                writer.add_scalar("Val/acc", val_acc.mean(), self.iteration)
                
                if self.config.report_ray:
                    tune.report(loss=val_loss.mean(), accuracy=val_acc.mean(), iteration=self.iteration)
                    sys.stdout.flush()
                
                if val_acc.mean() > max_acc:
                    max_acc = val_acc.mean()
                    best_model_epoch = self.iteration
                    best_model_state = self.model.state_dict()
                    
                self.log_history('val', self.iteration, val_loss, val_acc)
                self.save(_save_ckpt=False)

            if self.config.test_set_validation_freq:
                if (
                    self.iteration % self.config.test_set_validation_freq
                    == self.config.test_set_validation_freq - 1
                ):
                    logger.info("Validating on test set...")
                    _, test_acc, _ = self.model.eval_loop(self.test_loader)
                    writer.add_scalar("Test/acc", test_acc.mean(), self.iteration)

        logger.info(f"Training over after {self.config.n_epochs} epochs")
        logger.info("Retrieving model with best validation accuracy...")
        self.model.load_state_dict(best_model_state)
        logger.info(f"Retrieved model from epoch {best_model_epoch}")

        writer.close()

        return self.model

    def eval_model(self, model, save=True):
        logger.info("Initializing test data...")
        test_loader, test_dataset = get_episodic_loader(
            split="test",
            n_way=self.config.n_way_eval,
            n_source=self.config.n_source_eval,
            n_target=self.config.n_target,
            n_episodes=self.config.n_tasks_eval,
            no_change_perturb_s=self.config.no_change_perturb_test_s,
            no_change_perturb_q=self.config.no_change_perturb_test_s,
            support_query_shift=self.config.test_support_query_shift,
            image_size=self.config.image_size,
            dataset=self.config.dataset,
            data_dir=self.config.DATA_DIR,
            spec_file=self.config.TEST_SPEC_FILE
        )

        logger.info("Starting model evaluation...")
        model.eval()

        loss, acc, stats_df = model.eval_loop(test_loader)
        
        self.log_history("test", self.iteration, loss, acc)

        if save:
            stats_df = elucidate_ids(stats_df, test_dataset)
            stats_df.to_csv(os.path.join(self.config.EXP_DIR, "evaluation_stats.csv"), index=False)
            writer = SummaryWriter(log_dir=self.config.EXP_DIR)
            writer.add_scalar("Evaluation accuracy", acc.mean())
            writer.close()
            self.save()
        return acc


def load_model_episodic(model: nn.Module, state_dict: OrderedDict) -> nn.Module:
    model.load_state_dict(state_dict)
    return model


def load_model_non_episodic(config,
    model: nn.Module, state_dict: OrderedDict, use_fc: bool
) -> nn.Module:
    if use_fc:
        model.feature.trunk.fc = set_device(
            nn.Linear(
                model.feature.final_feat_dim,
                config.CLASSES["train"] + config.dataset_config.CLASSES["val"],
            )
        )

    model.feature.load_state_dict(
        state_dict
        if use_fc
        else OrderedDict([(k, v) for k, v in state_dict.items() if ".fc." not in k])
    )
    return model


def load_model(config,
    state_path, episodic: bool, use_fc: bool, force_ot: bool
) -> nn.Module:
    model = set_device(config.model(config.backbone))

    if force_ot:
        model.transportation_module = config.TRANSPORTATION_MODULE
        logger.info("Forced the Optimal Transport module into the model.")

    state_dict = torch.load(state_path)
    model = (
        load_model_episodic(config, model, state_dict)
        if episodic
        else load_model_non_episodic(model, state_dict, use_fc)
    )

    logger.info(f"Loaded model from {state_path}")

    return model