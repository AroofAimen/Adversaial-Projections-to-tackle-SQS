import numpy as np

class Config:
    def __init__(self):
        self.exp = "default"
        self.mode = 'train'
        # self.seed = np.random.randint(0, 2 ** 32 - 1)
        self.seed = 1
        self.image_size = 84
        self.gpu = 0
        self.report_ray = False
        self.val_freq = 200
        self.overwrite = False
        self.resume = False
        self.ckpt_no = None
        self.ckpt_start = 0
        self.ckpt_end = 0
        self.filter_size = 64

        # data config
        self.dataset = "mini_imagenet"
        self.DATA_DIR = "data"
        self.train_spec_file = "train.json"
        self.val_spec_file = "val.json"
        self.test_spec_file = "test.json"

        self.n_way = 5
        self.n_source = 5       # number of source samples per class
        self.n_target = 16      # number of target samples per class
        self.n_episodes = 4     
        self.n_val_tasks = 300  
        self.n_epochs = 10000
        # test tasks config
        self.n_way_eval = 5
        self.n_source_eval = 5
        self.n_target_eval = 5
        self.n_tasks_eval = 2000

        self.test_set_validation_freq = 0

        # domain config
        self.no_change_perturb_train_s = False
        self.no_change_perturb_train_q = False 
        self.no_change_perturb_test_s = False
        self.no_change_perturb_test_q = False

        self.test_support_query_shift = True
        self.train_support_query_shift = False
        
        self.clean_test_query = False
        
        self.use_test_query = False
        self.use_train_query = False

        # model config
        self.model_name = "ProtoNet"
        self.backbone = "conv4"
        self.use_elu = False
        self.base_lr = 0.5
        self.meta_lr = 0.003            # set this learning rate param for protonet
        self.weight_decay = 0.0001

        # transportation mod config
        self.use_transport_mod = False
        self.regularization = 0.05
        self.learn_regularization = False
        self.transport_max_iter = 1000
        self.stopping_criterion = 1e-4
        self.opt_transport_prob = 1.0
        
        # Adversarial Projection config
        self.use_adv_project = False
        self.adv_max_iter = 5
        self.adv_lr = 80.0
        self.adv_rand_conv_prob = 0.5
        self.train_project_task = 1       # 0 -> only support, 1 -> only query, 2 -> both support & query 
        self.test_project_task = 1       # 0 -> only support, 1 -> only query, 2 -> both support & query 
        self.adv_proj_prob = 0.5    # probability to use adversarially perturbed data for training
        self.plot_perturbed = False
        self.adv_lr_gamma = 1.0001
        self.adv_lr_max = 101
        self.adv_lr_sched = False