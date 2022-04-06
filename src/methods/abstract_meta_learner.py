from abc import abstractmethod
from audioop import bias

from loguru import logger
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.methods.utils import confidence_interval
from src.utils import set_device, plot_task

from src.adv_train import RCNN, modify_data


class AbstractMetaLearner(nn.Module):
    """
    Abstract class for meta-learning models. Extensions of this class must define the set_forward function.
    """

    def __init__(self,
                 model_func, 
                 transportation=None, 
                 use_test_query=False,
                 use_train_query=False,
                 training_stats=None,
                 adv_train_config=None,
                 **args
                 ):
        """

        Args:
            model_func (src.backbones object): backbone model initialized
            training_stats (Statistics): training statistics of the model, updated during training

        """
        super(AbstractMetaLearner, self).__init__()
        # importing inside class to avoid cyclic import
        from src.adv_train import transductive_adv_loss

        self.args = args
        self.report_ray = args.get("report_ray", False)
        self.n_episodes = args.get('n_episodes')
        self.feature = model_func
        self.training_stats = training_stats
        self.transportation_module = transportation
        self.use_test_query = use_test_query
        self.use_train_query = use_train_query
        self.loss_fn = nn.CrossEntropyLoss()
        self.adv_train_config = adv_train_config
        
        if self.use_test_query and (self.adv_train_config is not None):
            self.transductive_loss_fn = transductive_adv_loss
        else:
            self.transductive_loss_fn = None

        if 'ANIL' in args.get("model_name"):
            self.feature.cls = set_device(
                                nn.Linear(self.feature.final_feat_dim, args.get("n_way"), bias=True)
                                )
            self.meta_learner = set_device(self.build_metalearner(args.get("base_lr")))

    @abstractmethod
    def set_forward(self, support_images, support_labels, query_images):
        """
        Predict query set labels using information from support set labelled images.
        Must be implemented in all classes extending AbstractMetaLearner.
        Args:
            support_images (torch.Tensor): shape (number_of_support_set_images, **image_shape)
            support_labels (torch.Tensor): artificial support set labels in range (0, n_way)
            query_images (torch.Tensor): shape (number_of_query_set_images, **image_shape)

        Returns:
            torch.Tensor: shape(n_query*n_way, n_way), classification prediction for each query data
        """
        pass

    def fit_on_task(
        self, support_images, support_labels, 
        query_images, query_labels, 
        optimizer,
    ):
        """
        Perform a forward pass and a backward pass on one episode.
        Args:
            support_images (torch.Tensor): shape (number_of_support_set_images, **image_shape)
            support_labels (torch.Tensor): artificial support set labels in range (0, n_way)
            query_images (torch.Tensor): shape (number_of_query_set_images, **image_shape)
            query_labels (torch.Tensor): artificial query set labels in range (0, n_way)
            optimizer (torch.optim.Optimizer): model optimizer

        Returns:
            tuple(torch.Tensor, torch.Tensor): detached from the computational graph
                - shape(n_query*n_way, n_way), classification prediction for each query data
                - shape(,), training loss
        """
        optimizer.zero_grad()
        scores = self.set_forward(support_images, support_labels, query_images)
        loss = self.loss_fn(scores, query_labels)
        loss.backward()
        optimizer.step()

        return scores.detach(), loss.detach().item()

    def extract_features(self, support_images, query_images, feature_ext=None):
        """
        Computes the features vectors of the support and query sets
        Args:
            support_images (torch.Tensor): shape (n_support_images, **image_dim) input data
            query_images (torch.Tensor): shape (n_query_images, **image_dim) input data

        Returns:
            Tuple(torch.Tensor, torch.Tensor): features vectors of the support and query sets, respectively of shapes
            (n_support_images, features_dim) and (n_query_images, features_dim)
        """
        # Set to CUDA if available
        support_images = set_device(support_images)
        query_images = set_device(query_images)
        
        if feature_ext is not None:
            z_support = feature_ext.forward(support_images)
            z_query = feature_ext.forward(query_images)
        else:
            z_support = self.feature.forward(support_images)
            z_query = self.feature.forward(query_images)

        return z_support, z_query

    @staticmethod
    def get_prototypes(features, labels):
        """
        Compute a prototype for each class.
        Args:
            features (torch.Tensor): shape[n_images, feature_dim], feature vectors
            labels (torch.Tensor): shape[n_images], label corresponding to each feature vector
        Returns:
            torch.Tensor: prototypes. shape[number_of_unique_values_in_labels, feature_dim]
        """

        n_way = len(torch.unique(labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        return torch.cat(
            [features[torch.nonzero(labels == label)].mean(0) for label in range(n_way)]
        )

    @staticmethod
    def evaluate(scores, query_labels):
        """
        Predict labels of query images and returns the number of correct top1 predictions
        Args:
            scores (torch.Tensor): shape (number_of_query_set_images, n_way)
            query_labels (torch.Tensor): artificial query set labels in range (0, n_way)

        Returns:
            float: classification accuracy in [0,1]
        """

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        top1_correct = float((topk_labels[:, 0] == query_labels).sum())
        return float(top1_correct) / len(query_labels)

    @staticmethod
    def get_task_perf(
        task_id, classification_scores, labels, class_ids, source_id, target_id
    ):
        """
        Records the classification results for each query instance.
        Args:
            task_id (int): index of the task
            classification_scores (torch.Tensor): predicted classification scores
            labels (torch.Tensor): ground truth labels
            class_ids (list[int]): indices of the classes composing the current classification task
            source_id (int): index of the source domain for this task
            target_id (int): index of the target domain for this task

        Returns:
            pd.DataFrame: for each couple (query, class), gives classification score, class_id,
                ground truth query label, current task id and source and target domain,
        """
        return pd.concat(
            [
                pd.DataFrame(
                    {
                        "task_id": task_id,
                        "source_domain": source_id,
                        "target_domain": target_id,
                        "image_id": i,
                        "true_label": int(class_ids[labels[i]]),
                        "predicted_label": [
                            int(class_ids[label])
                            for label in range(classification_scores.shape[1])
                        ],
                        "score": classification_scores[i],
                    }
                )
                for i in range(labels.shape[0])
            ]
        )

    def plot_adv_task(self, adv_tasks, epoch):
        NUM_TASKS_TO_PLOT = 15
        episodes = list(adv_tasks.keys())
        try:
            episodes_to_plot = np.round(np.linspace(0, len(episodes)-1, NUM_TASKS_TO_PLOT)).astype(int)
        except:
            episodes_to_plot = np.round(np.linspace(0, len(episodes)-1, 10)).astype(int)
        for idx in episodes_to_plot:
            data = adv_tasks[episodes[idx]]
            if "query_before" and "query_after" in data:
                key_before = "query_before"
                title_before = "Query - Before Adversarial Projection"
                key_after = "query_after"
                title_after = "Query - After Adversarial Projection"
            elif "support_before" in data and "support_after" in data:
                key_before = "support_before"
                title_before = "Support - Before Adversarial Projection"
                key_after = "support_after"
                title_after = "Support - After Adversarial Projection"

            plot_task(data[key_before][0], data[key_before][1],
                        self.args["plot_path"], epoch, idx, key_before,
                        data["nrow"], data["ncol"], title_before
                        )
            plot_task(data[key_after][0], data[key_after][1],
                        self.args["plot_path"], epoch, idx, key_after,
                        data["nrow"], data["ncol"], title_after
                        )

    def adv_project(self, mode, episode_index, support_images, support_labels, query_images, query_labels, loss_fn):
        adv_tasks = {}
        if self.adv_train_config["{}_project_task".format(mode)] != 0:
            if self.args.get("plot_perturbed", False):
                adv_tasks[episode_index] = {"query_before": (query_images, query_labels),
                                            "nrow" : self.args["n_way"],
                                            "ncol" : self.args["n_target"]
                                            }

            query_images = RCNN(query_images, self.adv_train_config)
            query_images = modify_data(self, 
                                    support_images,
                                    support_labels,
                                    query_images,
                                    query_labels,
                                    loss_fn,
                                    self.adv_train_config | {'project_task': self.adv_train_config["{}_project_task".format(mode)]}
                                    )
            
            if self.args.get("plot_perturbed", False):
                adv_tasks[episode_index]["query_after"] = (query_images, query_labels)
    

        if self.adv_train_config["{}_project_task".format(mode)] != 1:  
            if self.args.get("plot_perturbed", False):
                adv_tasks[episode_index] = {"support_before": (support_images, query_labels),
                                            "nrow" : self.args["n_way"],
                                            "ncol" : self.args["n_source"]
                                            }
            
            support_images = RCNN(support_images, self.adv_train_config)
            support_images = modify_data(self, 
                                        support_images,
                                        support_labels,
                                        query_images,
                                        query_labels,
                                        loss_fn,
                                        self.adv_train_config | {'project_task': self.adv_train_config["{}_project_task".format(mode)]}
                                        )

            if self.args.get("plot_perturbed", False):
                adv_tasks[episode_index]["support_after"] = (support_images, query_labels)
                
            # update adv learning rate
            if self.adv_train_config.get("adv_lr_sched", None) is not None: 
                self.adv_train_config["lr"] = self.adv_train_config["adv_lr_sched"].step(
                                                                self.adv_train_config["lr"]
                                                            )

        return support_images, support_labels, query_images, query_labels, adv_tasks


    def train_loop(self, epoch, train_loader, optimizer):
        """
        Executes one training epoch
        Args:
            epoch (int): current epoch
            train_loader (DataLoader): loader of a given number of episodes
            optimizer (torch.optim.Optimizer): model optimizer

        Returns:
            tuple(float, float): resp. average loss and classification accuracy
        """
        print_freq = self.n_episodes
        adv_tasks = {}
        loss_list = []
        acc_list = []
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            _,
            _,
            _,
        ) in enumerate(train_loader):
            query_labels = set_device(query_labels)
            support_labels = set_device(support_labels)

            
            adv_prob = np.random.random()
            if self.use_train_query and \
              (self.adv_train_config is not None) and \
              (adv_prob > 1.0-self.args.get("adv_proj_prob", 0.5)):
                loss_fn = self.loss_fn
                support_images, support_labels, query_images, query_labels, adv_tasks = self.adv_project(
                    'train', episode_index, support_images, support_labels, query_images, query_labels, loss_fn
                )
                

            self.train()
            
            scores, loss_value = self.fit_on_task(
                support_images, support_labels,
                query_images, query_labels,
                optimizer, 
            )

            loss_list.append(loss_value)

            acc_list.append(self.evaluate(scores, query_labels) * 100)

            if episode_index % print_freq == print_freq - 1:
                logger.info(
                    "Epoch {epoch} | Batch {episode_index}/{n_batches} | Accuracy {acc} | Loss {loss}".format(
                        epoch=epoch,
                        episode_index=episode_index + 1,
                        n_batches=len(train_loader),
                        loss=np.asarray(loss_list).mean(),
                        acc=np.asarray(acc_list).mean()
                    )
                )
        if self.args.get("plot_perturbed", False) and len(adv_tasks):
            self.plot_adv_task(adv_tasks, epoch)
        return np.asarray(loss_list), np.asarray(acc_list)

    def eval_loop(self, test_loader):
        """

        Args:
            test_loader (DataLoader): loader of a given number of episodes

        Returns:
            tuple(float, float, pd.DataFrame): resp. average loss and classification accuracy,
                and advanced evaluation statistics
        """
        acc_all = []
        loss_all = []
        evaluation_stats = []

        n_tasks = len(test_loader)
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            class_ids,
            source_domain,
            target_domain,
        ) in enumerate(test_loader):

            query_labels = set_device(query_labels)
            support_labels = set_device(support_labels)


            if self.use_test_query and \
              (self.adv_train_config is not None):
                loss_fn = self.transductive_loss_fn
                support_images, support_labels, query_images, query_labels, adv_tasks = self.adv_project(
                    'test', episode_index, support_images, support_labels, query_images, query_labels, loss_fn
                )

            scores = self.set_forward(
                support_images, support_labels, query_images
            ).detach()

            loss_value = self.loss_fn(scores, query_labels).detach().item()

            evaluation_stats.append(
                self.get_task_perf(
                    episode_index,
                    scores.cpu(),
                    query_labels.cpu().detach(),
                    class_ids,
                    source_domain,
                    target_domain,
                )
            )

            loss_all.append(loss_value)

            acc_all.append(self.evaluate(scores, query_labels) * 100)


        evaluations_stats_df = pd.concat(evaluation_stats, ignore_index=True)
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        logger.info(
            "%d Test Accuracy = %4.2f%% +- %4.2f%%"
            % (n_tasks, acc_mean, confidence_interval(acc_std, n_tasks))
        )

        return np.asarray(loss_all), acc_all, evaluations_stats_df