import json
import os
import argparse
from loguru import logger
from shutil import rmtree
from src.config import Config

import torch

def prepare_output(config):
    if config.overwrite & os.path.isdir(config.EXP_DIR):
        rmtree(str(config.EXP_DIR))
        logger.info(
            "Deleting previous content of {directory}",
            directory=config.EXP_DIR,
        )
    
    os.makedirs(config.EXP_DIR, exist_ok=True)
    logger.info(
        "Parameters and outputs of this experiment will be saved in {directory}",
        directory=config.EXP_DIR,
    )



def parse_args():
    config = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg', type=json.loads)
    exp_config = parser.parse_args().arg
    logger.info(exp_config)
    for param in exp_config:
        if hasattr(config, param):
            print(" Overriding Parameter : {} \t Initial Value : {} \t New Value : {}".format(
                param,config.__dict__[param],exp_config[param])
                )
            setattr(config,param,exp_config[param])
        else:
            print(" Unknown Parameter : {} ".format(param))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    config.dev  = torch.device("cuda:{}".format(0)) 
    
    EXP_DIR  = "./exp/{}/{}/SZ_{}/{}.{}.{}/{}_F/{}/{}/adv_{}_optTransport_{}/change_s_{}_change_q_{}/train_s_{}_train_q_{}_test_s_{}_test_q_{}/train_sq_shift_{}/test_sq_shift_{}".format(
            config.exp, config.dataset, config.image_size,
            config.n_way, config.n_source,
            config.n_target, config.filter_size,
            config.model_name, config.backbone,
            int(config.use_adv_project), int(config.use_transport_mod),
            int(config.train_project_task != 1), int(config.train_project_task != 0),
            int(1-config.no_change_perturb_train_s), int(1-config.no_change_perturb_train_q),
            int(1-config.no_change_perturb_test_s), int(1-config.no_change_perturb_test_q),
            int(config.train_support_query_shift), int(config.test_support_query_shift)
            )
    SPEC_ROOT = "./configs/dataset_specs/{}".format(config.dataset)
    DATA_DIR = "~/data/{}/".format(config.dataset)

    config.EXP_DIR  = EXP_DIR
    config.CKPT_DIR = os.path.join(EXP_DIR, "ckpt")
    config.LOG_DIR =  os.path.join(EXP_DIR, "logs")
    config.DATA_DIR = DATA_DIR
    
    os.makedirs(config.CKPT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    config.TRAIN_SPEC_FILE = os.path.join(SPEC_ROOT,config.train_spec_file)
    config.VAL_SPEC_FILE = os.path.join(SPEC_ROOT,config.val_spec_file)
    config.TEST_SPEC_FILE = os.path.join(SPEC_ROOT,config.test_spec_file)
    prepare_output(config)
    
    return config