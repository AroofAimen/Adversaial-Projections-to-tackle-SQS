import numpy as np
import random
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from loguru import logger

from src.utils import set_device

class Adv_lr_schedular:
    def __init__(self, gamma, max_lr):
        self.gamma = gamma
        self.max_lr = max_lr
    
    def step(self, lr):
        if self.gamma < self.max_lr:
            self.gamma *= self.gamma
        return min(lr*self.gamma, self.max_lr)


def RCNN(X, params):
    """
    Apply Random Convolution on Data

    Args:
        X (tensor): Dataset (X_support, X_query)
        params: configuration 
    """
    N, C, H, W = X.size()
    # K = [1, 3, 5, 7, 11, 15]
    K = [1, 3, 5]
    # K = [1, 3]
    if np.random.rand() > params["rand_conv_prob"]:
        k = random.choice(K)
        Conv = nn.Conv2d(3, 3, kernel_size=k, stride=1, padding=k//2, bias=False)
        nn.init.xavier_normal_(Conv.weight)
        X = Conv(X.reshape(-1, C, H, W)).reshape(N, C, H, W)
    return X.detach()

def modify_data(model,
                support_images,
                support_labels,
                query_images,
                query_labels,
                loss_fn,
                params,
                ):
    """
    Apply gradient ascent to modify input X

    Args:
        model (nn.Module): model
        X (tensor): input task
        params: configuration
    """
    support_images, support_labels, query_images, query_labels = set_device(
                            [support_images, support_labels, query_images, query_labels]
                            )
    if params["project_task"] == 0:
        optimizer = optim.SGD([support_images.requires_grad_(True)], lr = params["lr"])
        model.eval()
        for _ in range(params["max_T"]):
            optimizer.zero_grad()
            scores = model.set_forward(support_images, support_labels, query_images)
            loss = loss_fn(scores, query_labels)
            (-loss).backward()
            optimizer.step()
        
        return support_images.detach()
    
    elif params["project_task"] == 1:
        optimizer = optim.SGD([query_images.requires_grad_(True)], lr=params["lr"])
        model.eval()
        for _ in range(params["max_T"]):
            optimizer.zero_grad()
            scores = model.set_forward(support_images, support_labels, query_images)
            loss = loss_fn(scores, query_labels)
            (-loss).backward()
            optimizer.step()
            
        return query_images.detach()
    
    else:
        raise NotImplementedError("Can only project query or support individually yet.")
        

def transductive_adv_loss(query_scores, *kwargs):
    "entropy function on query scores"
    norm_scores = F.softmax(query_scores, 1)
    loss_val = (- norm_scores * torch.log(norm_scores + 1e-5)).sum(1).mean()
    return loss_val
