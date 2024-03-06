import numpy as np
from easydict import EasyDict
import utils


def coseg_params(type):              # aliens / chairs / vases

    sub_folder = 'coseg_' + type     # coseg_aliens
    p = 'datasets_processed/coseg_from_meshcnn/' + sub_folder + '/'  # 'datasets_processed/coseg_from_meshcnn/coseg_aliens/'
    params = set_up_default_params('semantic_segmentation', 'coseg_' + type, 0)
    params.n_classes = 10
    params.seq_len = 300
    params.min_seq_len = int(params.seq_len / 2)

    params.datasets2use['train'] = [p + '*train*.npz']  # ['datasets_processed/coseg_from_meshcnn/coseg_aliens/*train*.npz']
    params.datasets2use['test'] = [p + '*test*.npz']

    params.iters_to_train = 200e3  # 20万次迭代
    params.train_data_augmentation = {'rotation': 360}  # 数据增强策略

    params.full_accuracy_test = {'dataset_expansion': params.datasets2use['test'][0],
                                 'n_iters': 32}

    return params

















# Semantic Segmentation
# ---------------------
def human_seg_params():
    params = set_up_default_params('semantic_segmentation', 'human_seg', 0)
    params.n_classes = 9
    params.seq_len = 300
    params.min_seq_len = int(params.seq_len / 2)

    p = 'datasets_processed/human_seg_from_meshcnn/'
    params.datasets2use['train'] = [p + '*train*.npz']
    params.datasets2use['test'] = [p + '*test*.npz']

    params.train_data_augmentation = {'rotation': 360}

    params.full_accuracy_test = {'dataset_expansion': params.datasets2use['test'][0],
                                 'n_iters': 32}

    params.iters_to_train = 100e3

    return params





def psb_params(job):
    pass
    # params = set_up_default_params('semantic_segmentation', job, 0)
    # params.n_classes = 9
    # params.seq_len = 300
    # params.min_seq_len = int(params.seq_len / 2)
    #
    # p = 'datasets_processed/' + job + '/'
    # params.datasets2use['train'] = [p + '*train*.npz']
    # params.datasets2use['test'] = [p + '*test*.npz']
    #
    # params.train_data_augmentation = {'rotation': 360}
    #
    # params.full_accuracy_test = {'dataset_expansion': params.datasets2use['test'][0],
    #                              'n_iters': 32}
    #
    # params.iters_to_train = 70e3
    #
    # return params


def cosegs_params(job):
    params = set_up_default_params('semantic_segmentation', job, 0)
    params.n_classes = 4
    params.seq_len = 300
    params.min_seq_len = int(params.seq_len / 2)

    p = 'datasets_processed/' + job + '/'
    params.datasets2use['train'] = [p + '*train*.npz']
    params.datasets2use['test'] = [p + '*test*.npz']

    params.train_data_augmentation = {'rotation': 360}

    params.full_accuracy_test = {'dataset_expansion': params.datasets2use['test'][0],
                                 'n_iters': 32}

    params.iters_to_train = 80e3

    return params
