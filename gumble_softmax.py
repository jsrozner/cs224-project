import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util

from args import get_train_args
from collections import OrderedDict
from json import dumps
import json
from models import BiDAF
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD

import tensorflow as tf



def sampleGumbel(shape, eps=1e-20):
  """Sample from Gumbel distribution(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)


def gumbelSoftmaxSample(log, temp):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = log + sampleGumbel(tf.shape(log))
  return tf.nn.softmax( y / temp)


def gumbelSoftmax(log, temp, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbelSoftmaxSample(log, temp)
  if hard:
    k = tf.shape(log)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y


def openJson(file_name):
    with open(file_name) as json_file:
        data = json.load(json_file)

    return data


def paraphraseModel(paraphrases, device, use_squad=True):

    paraphrases["data"] =

    context_idxs = torch.from_numpy(np_context_idxs).long()
    cw_idxs = context_idxs.to(device)

    if use_squad:
        batch_size, c_len, w_len = context_idxs.size()
        ones = torch.ones((batch_size, 1), dtype=torch.int64)
        context_idxs = torch.cat((ones, self.context_idxs), dim=1)

    # Get embeddings
    word_vectors = util.torch_from_json("data/word_emb.json")
    word2idx = openJson("data/word2idx.json")

    for p in paraphrases:





if __name__=="__main__":
    # print(openJson("data/word2idx.json"))
    paraphraseModel()
