from posixpath import split
import random

from loguru import logger
import torch
from torch.utils.data import Sampler

from sklearn.model_selection import train_test_split


class BeforeCorruptionSampler(Sampler):
    """
    Sample images from a dataset which uses image perturbations (like CIFAR100-C or tieredImageNet-C), in the
    case where perturbations are applied online. For the case where images on disk are already corrupted, see
    AfterCorruptionSampler.
    """

    def __init__(self, dataset, n_way, n_source, n_target, n_episodes, **args):
        self.dataset = dataset
        self.n_domains = len(dataset.id_to_domain)
        # logger.info("{}:\t{}\t{}\t".format(dataset.split, self.n_domains, dataset.id_to_domain))
        # logger.info("{}:\t{}\t".format(dataset.split, dataset.perturbations))
        # logger.info("no_change perturbation: {}".format(self._get_no_change_perturbation()))
        self.n_total_images = len(dataset.images)
        self.n_way = n_way
        self.n_source = n_source
        self.n_target = n_target
        self.n_episodes = n_episodes
        self.args = args

        self.items_per_label = {}

        for item, label in enumerate(dataset.labels):
            if label in self.items_per_label.keys():
                self.items_per_label[label].append(item)
            else:
                self.items_per_label[label] = [item]

    def __len__(self):
        return self.n_episodes

    def _split_source_target(self, labels):
        source_items_per_label = {}
        target_items_per_label = {}

        for label in labels:
            (
                source_items_per_label[label],
                target_items_per_label[label],
            ) = train_test_split(self.items_per_label[label], train_size=0.5)
        return source_items_per_label, target_items_per_label

    @staticmethod
    def _sample_instances(items, n_samples):
        return torch.tensor(
            items if n_samples == -1 else random.sample(items, n_samples)
        )

    # UPDATE
    def _get_no_change_perturbation(self):
        idx = -1
        for i, fn in enumerate(self.dataset.perturbations):
            if fn.func.__name__ == 'no_change':
                idx = i
                break
        return idx

    def _get_episode_items(self):
        labels = random.sample(self.items_per_label.keys(), self.n_way)

        source_items_per_label, target_items_per_label = self._split_source_target(
            labels
        )

        no_change_perturb = self._get_no_change_perturbation()


        if self.args.get('support_query_shift', False):
            if self.args.get('no_change_perturb_s', False):
                source_perturbation = no_change_perturb
                if self.args.get('no_change_perturb_q', False):
                    target_perturbation = no_change_perturb
                else:
                    target_perturbation = torch.tensor(
                                                random.choice([
                                                    i for i in range(self.n_domains) if i != source_perturbation
                                                ])
                                            )
            else:
                source_perturbation = torch.tensor(
                                                random.choice([
                                                    i for i in range(self.n_domains) if i != no_change_perturb
                                                ])
                                            )
                if self.args.get('no_change_perturb_q', False):
                    target_perturbation = no_change_perturb
                else:
                    target_perturbation = torch.tensor(
                                                random.choice([
                                                    i for i in range(self.n_domains) if i != source_perturbation
                                                ])
                                            )
            
        else:
            if self.args.get('no_change_perturb_s', False) or \
                self.args.get('no_change_perturb_q', False):
                source_perturbation = no_change_perturb
                target_perturbation = no_change_perturb
            else:
                source_perturbation = torch.tensor(
                                                random.choice([
                                                    i for i in range(self.n_domains) if i != no_change_perturb
                                                ])
                                            )
                target_perturbation = source_perturbation


        source_items = (
            torch.cat(
                [
                    self._sample_instances(source_items_per_label[label], self.n_source)
                    for label in labels
                ]
            )
            * self.n_domains
            + source_perturbation
        )

        target_items = (
            torch.cat(
                [
                    self._sample_instances(target_items_per_label[label], self.n_target)
                    for label in labels
                ]
            )
            * self.n_domains
            + target_perturbation
        )

        return torch.cat((source_items, target_items))

    def __iter__(self):
        for _ in range(self.n_episodes):
            yield self._get_episode_items()
