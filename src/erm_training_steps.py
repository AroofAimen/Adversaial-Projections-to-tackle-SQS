"""
Steps used in scripts/erm_training.py
"""

import sys
from typing import OrderedDict
from ray import tune

from loguru import logger
import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, random_split, DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from configs import dataset_config, erm_training_config, experiment_config, model_config
from src.data_tools.datasets import _DATASETS
from src.utils import get_episodic_loader, load_ckpt, save_ckpt, save_history, set_device


class ERM_Trainer:
    def __init__(self, config):
        self.config = config
        self.train_loader, self.val_loader, self.n_classes = self.get_data()
        self.get_model(self.n_classes)

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
        

    def get_few_shot_split(self) -> (Dataset, Dataset):
        temp_train_set = _DATASETS[self.config.dataset](
            self.config.DATA_DIR, "train", self.config.image_size
        )
        temp_train_classes = len(temp_train_set.id_to_class)
        temp_val_set = _DATASETS[self.config.dataset](
            self.config.DATA_DIR,
            "val",
            self.config.image_size,
            target_transform=lambda label: label + temp_train_classes,
        )
        if hasattr(_DATASETS[self.config.dataset], "__name__"):
            if _DATASETS[self.config.dataset].__name__ == "CIFAR100CMeta":
                label_mapping = {
                    v: k
                    for k, v in enumerate(
                        list(temp_train_set.id_to_class.keys())
                        + list(temp_val_set.id_to_class.keys())
                    )
                }
                temp_train_set.target_transform = (
                    temp_val_set.target_transform
                ) = lambda label: label_mapping[label]

        return temp_train_set, temp_val_set


    def get_non_few_shot_split(
        self, temp_train_set: Dataset, temp_val_set: Dataset
    ) -> (Subset, Subset):
        train_and_val_set = ConcatDataset(
            [
                temp_train_set,
                temp_val_set,
            ]
        )
        n_train_images = int(
            len(train_and_val_set) * self.config.ERM_train_images_proportion
        )
        return random_split(
            train_and_val_set,
            [n_train_images, len(train_and_val_set) - n_train_images],
            generator=torch.Generator().manual_seed(
                self.config.ERM_train_val_split_random_seed
            ),
        )


    def get_data(self) -> (DataLoader, DataLoader, int):
        logger.info("Initializing data loaders...")

        temp_train_set, temp_val_set = self.get_few_shot_split()

        train_set, val_set = self.get_non_few_shot_split(temp_train_set, temp_val_set)

        train_loader = DataLoader(
            train_set,
            batch_size=self.config.ERM_batch_size,
            num_workers=self.config.ERM_n_workers,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.config.ERM_batch_size,
            num_workers=self.config.ERM_n_workers,
        )
        # Assume that train and val classes are entirely disjoints
        n_classes = len(temp_val_set.id_to_class) + len(temp_train_set.id_to_class)

        return train_loader, val_loader, n_classes


    def get_model(self, n_classes: int) -> nn.Module:
        logger.info(f"Initializing {model_config.BACKBONE.__name__}...")
        model = set_device(self.config.BACKBONE())
        model.trunk.add_module("fc", set_device(nn.Linear(model.final_feat_dim, n_classes)))
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = self.config.optimizer(model.parameters())
        self.model = model


    def get_n_batches(self, data_loader: DataLoader, n_images_per_epoch: int) -> int:
        """
        Computes the number of batches in a training epoch from the intended number of seen images.
        """

        return min(n_images_per_epoch // self.config.ERM_batch_size, len(data_loader))


    def fit(self, images: torch.Tensor, labels: torch.Tensor):
        self.optimizer.zero_grad()
        scores = self.model(images)
        loss = self.loss_fn(scores, labels)
        loss.backward()
        self.optimizer.step()

        acc = float(
                    (
                        scores.data.topk(1, 1, True, True)[1][:, 0]
                        == labels
                    )
                ).sum() / len(labels)

        return acc.item(), loss.item()


    def training_epoch(
        self, data_loader: DataLoader, epoch: int, n_batches: int
    ):
        loss_list = []
        acc_list = []
        self.model.train()

        with tqdm(
            zip(range(n_batches), data_loader),
            total=n_batches,
            desc=f"Epoch {epoch}",
        ) as tqdm_train:
            for batch_id, (images, labels, _) in tqdm_train:
                acc_value, loss_value = self.fit(set_device(images), set_device(labels))
                loss_list.append(loss_value)
                acc_list.append(acc_value)
                tqdm_train.set_postfix(loss=np.asarray(loss_list).mean())

        return np.asarray(acc_list), np.asarray(loss_list)


    def validation(self, data_loader: DataLoader, n_batches: int):
        val_acc_list, val_loss_list = [], []
        self.model.eval()
        with tqdm(
            zip(range(n_batches), data_loader),
            total=n_batches,
            desc="Validation:",
        ) as tqdm_val:
            for _, (images, labels, _) in tqdm_val:
                scores = self.model(set_device(images))
                loss = self.loss_fn(scores, set_device(labels))
                acc = float(
                        (
                            scores.data.topk(1, 1, True, True)[1][:, 0]
                            == set_device(labels)
                        ).sum()
                    ) / len(labels)

                val_acc_list.append(acc)
                val_loss_list.append(loss)
                tqdm_val.set_postfix(accuracy=np.asarray(val_acc_list).mean())

        return np.asarray(val_acc_list), np.asarray(val_loss_list)


    def train(self) -> (OrderedDict, int):
        writer = SummaryWriter(log_dir=self.config.EXP_DIR)
        n_training_batches = self.get_n_batches(
            self.train_loader, self.config.ERM_n_train_images_per_epoch
        )
        n_val_batches = self.get_n_batches(
            self.val_loader, self.config.ERM_n_val_images_per_epoch
        )
        max_val_acc = 0.0
        best_model_epoch = 0
        best_model_state = self.model.state_dict()
        logger.info("Model and data are ready. Starting training...")
        for epoch in range(self.config.ERM_n_epochs):

            train_acc, train_loss = self.training_epoch(
                self.model, self.train_loader, epoch, n_training_batches
            )

            writer.add_scalar(
                "Train/loss",
                train_loss.mean(),
                epoch,
            )

            val_acc, val_loss = self.validation(self.model, self.val_loader, n_val_batches)
            writer.add_scalar("Val/acc", val_acc, epoch)

            if self.config.report_ray:
                tune.report(loss=val_loss.mean(), accuracy=val_acc.mean(), iteration=self.iteration)
                sys.stdout.flush()

            if val_acc.mean() > max_val_acc:
                max_val_acc = val_acc.mean()
                best_model_epoch = epoch
                best_model_state = self.model.state_dict()

            self.log_history('train', epoch, train_loss, train_acc)
            self.log_history('val', epoch, train_loss, train_acc)

        return best_model_state, best_model_epoch


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
        
        # self.log_history('val', self.iteration, loss, acc)

        self.log_history("test", self.iteration, loss, acc)

        if save:
            stats_df = elucidate_ids(stats_df, test_dataset)
            stats_df.to_csv(os.path.join(self.config.EXP_DIR, "evaluation_stats.csv"), index=False)
            writer = SummaryWriter(log_dir=self.config.EXP_DIR)
            writer.add_scalar("Evaluation accuracy", acc.mean())
            writer.close()
            self.save()
        return acc


def wrap_up_training(best_model_state: OrderedDict, best_model_epoch: int):
    logger.info(f"Training complete.")
    logger.info(f"Best model found after {best_model_epoch + 1} training epochs.")
    state_dict_path = (
        experiment_config.SAVE_DIR
        / f"{model_config.BACKBONE.__name__}_{dataset_config.DATASET.__name__ if hasattr(dataset_config.DATASET, '__name__') else dataset_config.DATASET.func.__name__}.tar"
    )
    torch.save(best_model_state, state_dict_path)
    logger.info(f"Model state dict saved in {state_dict_path}")
