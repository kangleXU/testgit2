import torch


from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy
from new_models import HAN, SpGAT
from new_preprocess import read_entity_from_id, read_relation_from_id, init_embeddings, build_data
from create_batch import Corpus
from utils import save_model

import random
import argparse
import os
import sys
import logging
import time
import pickle


def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-data", "--data", default="./data/WN18RR", help="data directory")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,default=True, help="Use pretrained embeddings")
    args.add_argument('--sparse', action='store_true', default=True, help='GAT with sparse version or not.')
    args.add_argument("-out_edim", "--entity_out_dim", type=int,default=100, help="Entity output embedding dimensions")
    args.add_argument("-out_rdim", "--relation_out_dim", type=int, default=100, help="relation output embedding dimensions")
    args.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    args.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    args.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    args.add_argument('--hidden_units', type=int, default=8, help='Number of head attentions.')
    args.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    args.add_argument("-w_conv", "--weight_decay_conv", type=float, default=1e-5, help="L2 reglarization for conv")
    args.add_argument("-margin", "--margin", type=float, default=5, help="Margin used in hinge loss")
    args.add_argument("-e_g", "--epochs_gat", type=int, default=3600, help="Number of epochs")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=50, help="Size of embeddings (if pretrained not used)")

    # ConvE specific hyperparameters
    args.add_argument('-hid_drop2', dest='hid_drop2', default=0.3, type=float, help='ConvE: Hidden dropout')
    args.add_argument('-feat_drop', dest='feat_drop', default=0.3, type=float, help='ConvE: Feature Dropout')
    args.add_argument('-k_w', dest='k_w', default=10, type=int, help='ConvE: k_w')
    args.add_argument('-k_h', dest='k_h', default=20, type=int, help='ConvE: k_h')
    args.add_argument('-num_filt', dest='num_filt', default=200, type=int,
                        help='ConvE: Number of filters in convolution')
    args.add_argument('-ker_sz', dest='ker_sz', default=7, type=int, help='ConvE: Kernel size to use')

    args.add_argument('-logdir', dest='log_dir', default='./log/', help='Log directory')
    args.add_argument('-config', dest='config_dir', default='./config/', help='Config directory')
    args.add_argument('-embed_dim', dest='embed_dim', default=200, type=int,
                        help='Embedding dimension to give as input to score function')
    args.add_argument('-inp_drop', dest="inp_drop", default=0.2, type=float, help='Dropout for full connected layer')
    # args.add_argument('-bias', dest='bias', action='store_true', help='Whether to use bias in the model')
    args.add_argument('-form', type=str, default='alternate', help='Input concat form')
    args = args.parse_args()
    return args

args = parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def load_data(args):
    train_data, validation_data, test_data, entity2id, relation2id = build_data(
        args.data, is_unweigted=False, directed=True)

    if args.pretrained_emb:
        entity_embeddings, relation_embeddings = init_embeddings(os.path.join(args.data, 'entity2vec.txt'),
                                                                 os.path.join(args.data, 'relation2vec.txt'))
        print("Initialised relations and entities from TransE")

    else:
        entity_embeddings = np.random.randn(
            len(entity2id), args.embedding_size)
        relation_embeddings = np.random.randn(
            len(relation2id), args.embedding_size)
        print("Initialised relations and entities randomly")
    entity_embeddings = torch.FloatTensor(entity_embeddings)
    relation_embeddings = torch.FloatTensor(relation_embeddings)
    print("Initial entity dimensions {} , relation dimensions {}".format(entity_embeddings.size(), relation_embeddings.size()))
    print(len(relation_embeddings))
    return train_data, validation_data, test_data, entity2id, relation2id, entity_embeddings, relation_embeddings

train_data, validation_data, test_data, entity2id, relation2id, entity_embeddings, relation_embeddings = load_data(args)
CUDA = torch.cuda.is_available()

def train_gat(args):
    model =  HAN(    num_meta_paths=len(relation_embeddings),
                     entity_embedd=entity_embeddings,
                     hidden_size = args.hidden_units,
                     relation_dim = args.embedding_size,
                     relation_embed= relation_embeddings,
                     entity_out_dim=args.entity_out_dim,
                     num_heads=args.nb_heads,
                     dropout=args.dropout,
                     alpha=args.alpha,
                     # conve c参数
                     num_filt = args.num_filt,
                     hid_drop2 = args. hid_drop2,
                     feat_drop  = args.feat_drop ,
                     k_w = args.k_w,
                     k_h = args.k_h,
                     ker_sz = args.ker_sz,
                     embed_dim = args.embed_dim,
                     inp_drop = args.inp_drop,
                     form = args.form,
                     bias = args.bias

                    )
    if CUDA:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5, last_epoch=-1)
    gat_loss_func = nn.MarginRankingLoss(margin=args.margin)

    epoch_losses = []  # losses of all epochs
    print("Number of epochs {}".format(args.epochs_gat))
    epoch = 0


if __name__ == '__main__':
    load_data(args)
    train_gat(args)