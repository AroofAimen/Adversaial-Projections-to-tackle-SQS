from distutils.command.config import config
import glob
import os
import copy
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.utils.data import DataLoader

from src.data_tools.utils import episodic_collate_fn_wrap
from src.data_tools.datasets import _DATASETS


def set_device(x, id=0):
    """
    Switch a tensor to GPU if CUDA is available, to CPU otherwise
    """
        
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(id))
    else:
        device = torch.device("cpu")

    if isinstance(x, list):
        return [item.to(device=device) 
                # if isinstance(item, torch.Tensor) else item
                for item in x]

    return x.to(device=device)

def plot_episode(support_images, query_images):
    """
    Plot images of an episode, separating support and query images.
    Args:
        support_images (torch.Tensor): tensor of multiple-channel support images
        query_images (torch.Tensor): tensor of multiple-channel query images
    """

    def matplotlib_imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    support_grid = torchvision.utils.make_grid(support_images)
    matplotlib_imshow(support_grid)
    plt.title("support images")
    plt.show()
    query_grid = torchvision.utils.make_grid(query_images)
    plt.title("query images")
    matplotlib_imshow(query_grid)
    plt.show()


def elucidate_ids(df, dataset):
    """
    Retrieves explicit class and domain names in dataset from their integer index,
        and returns modified DataFrame
    Args:
        df (pd.DataFrame): input DataFrame. Must be the same format as the output of AbstractMetaLearner.get_task_perf()
        dataset (Dataset): the dataset
    Returns:
        pd.DataFrame: output DataFrame with explicit class and domain names
    """
    return df.replace(
        {
            "predicted_label": dataset.id_to_class,
            "true_label": dataset.id_to_class,
            "source_domain": dataset.id_to_domain,
            "target_domain": dataset.id_to_domain,
        }
    )


def get_episodic_loader(
    split: str, 
    n_way: int,
    n_source: int,
    n_target: int,
    n_episodes: int,
    no_change_perturb_s: bool,
    no_change_perturb_q: bool,
    support_query_shift: bool,
    image_size: int,
    dataset,
    data_dir: str,
    spec_file,
):
    dataset = _DATASETS[dataset](
        data_dir,
        split,
        image_size,
        spec_file=spec_file,
    )
    sampler = dataset.get_sampler()(
        n_way=n_way,
        n_source=n_source,
        n_target=n_target,
        n_episodes=n_episodes,
        **{
            "no_change_perturb_s": no_change_perturb_s,
            "no_change_perturb_q": no_change_perturb_q,
            "support_query_shift": support_query_shift
        }
    )
    return (
        DataLoader(
            dataset,
            batch_sampler=sampler,
            # num_workers=12,
            num_workers=1,
            pin_memory=False,
            collate_fn=episodic_collate_fn_wrap(n_way, n_source, n_target),
        ),
        dataset,
    )


def save_ckpt(history, iteration, metalearner=None, optim=None, save="./"):
    os.makedirs(save,exist_ok=True)
    torch.save({
        'iteration': iteration,
        'model': metalearner.state_dict(),
        'optim':       optim.state_dict(),
        'history': history
    }, os.path.join(save, 'meta-learner-{}.pth.tar'.format(iteration)))
    

def load_ckpt(model, optim, device, ckpt_dir, ckpt_no=None, logger=None):
    list_of_files = glob.glob(os.path.join(ckpt_dir, '*'))
    if ckpt_no is None:
        latest_file = max(list_of_files, key=os.path.getmtime)
    else:
        latest_file = os.path.join(ckpt_dir,"meta-learner-{}.pth.tar".format(ckpt_no))
    print(ckpt_dir, latest_file)
    
    if logger is not None:
        logger.info("Resuming From : {}".format(latest_file))
    print("Resuming From : ", latest_file)
    ckpt = torch.load(latest_file, map_location=device)
    last_iteration          = ckpt['iteration']
    pretrained_state_dict = ckpt['model']
    
    model.load_state_dict(pretrained_state_dict)
    optim.load_state_dict(ckpt['optim'])
    history = ckpt['history']
    
    return model, optim, last_iteration, history


def save_history(history, name, log_dir):
    save_path = "{}/history.json".format(log_dir)
    with open(save_path, "wb") as file:
        pickle.dump(history, file)
    
    plot_learn(history, name, log_dir)
    

def load_history(log_dir):
    load_path = "{}/history.json".format(log_dir)
    with open(load_path, 'rb') as f:
        history = pickle.load(f)
    return history


def plot_learn(history, name, log_dir):
    plot_variables = ["loss", "accuracy"]
    sns.set_style("whitegrid")
    for idx, plot_key in enumerate(plot_variables):
        plt.figure(figsize=(16,8))
        plt.title("{}".format(name), fontsize=16)
        if "train"  in history[plot_key].keys() and len(history["iterations"]["train"])==len(history[plot_key]["train"]):
                plt.plot(history["iterations"]["train"],history[plot_key]["train"],label = "Training",color='b',linestyle='--',alpha=0.5)
            
        if "val"  in history[plot_key].keys() and len(history["iterations"]["val"])==len(history[plot_key]["val"]):
            plt.plot(history["iterations"]["val"],history[plot_key]["val"],label = "Validation",color='coral',linewidth=2)
    
        if plot_key+"-std" in history.keys():
            if "train"  in history[plot_key+"-std"].keys():
                plt.fill_between(history["iterations"]["train"],np.array(history[plot_key]["train"])+ np.array(history[plot_key+"-std"]["train"]),np.array(history[plot_key]["train"])- np.array(history[plot_key+"-std"]["train"]),color='b',alpha=0.25)
            
            if "val"  in history[plot_key+"-std"].keys():
                plt.fill_between(history["iterations"]["val"],np.array(history[plot_key]["val"])+np.array(history[plot_key+"-std"]["val"]),np.array(history[plot_key]["val"])-np.array(history[plot_key+"-std"]["val"]),color='coral',alpha=0.25)

        if "test"  in history[plot_key].keys():
                plt.axhline(np.mean(history[plot_key]["test"]),color='teal',linestyle='--',label='Testing',linewidth=3)


        plt.legend(fontsize=15)
        plt.xlabel('Iterations',fontsize=15)
        plt.ylabel(' {} '.format(plot_key) ,fontsize=15)
        plt.savefig("{}/{}.png".format(log_dir,plot_key))
        plt.close()


def tensor_to_npimg(img):
    image = img.clone().detach()
    npimg = np.transpose(image.cpu().numpy(), (1,2,0))
    npimg = (npimg*255).astype(np.uint8)
    return npimg


def plot_task(images, labels, path, epoch, task_idx, _set, nrow=4, ncol=5, title="set"):
    os.makedirs(path, exist_ok=True)

    imgs, lbls = [], []
    for item in range(images.shape[0]):
        imgs.append(tensor_to_npimg(images[item]))
        lbls.append(labels[item])

    figsize = (7, 7) if ncol == 5 else (20, 7)
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111, 
                nrows_ncols=(nrow, ncol),
                axes_pad=(0.,0.36)
            )

    for idx, (ax, img, lbl) in enumerate(zip(grid, imgs, lbls)):
        ax.imshow(img)
        if idx%ncol == 0:
            ax.set_ylabel("{}".format(lbls[idx]), rotation=90, fontsize=8, labelpad=20)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
    
    path2 = os.path.join(path, "_{}_{}_{}.png".format(epoch, task_idx, _set))
    plt.savefig("{}".format(path2))
    plt.close()