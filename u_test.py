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
from models import BiDAF
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD

from pprint import pprint as pp


def main(args):
    log = util.get_logger(args.save_dir, args.name)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)

    # Prepare BiDAF model (must already trained)
    log.info('Building BiDAF model (should be pretrained)')
    bidaf_model = BiDAF(word_vectors=word_vectors,          # todo: these word vectors shouldn't matter?
                          hidden_size=args.hidden_size)     # since they will be loaded in during load_model?
                          #drop_prob=args.drop_prob)        # no drop probability since we are not training
    #bidaf_model = nn.DataParallel(bidaf_model, args.gpu_ids)

    #log.info(f'Loading checkpoint from {args.load_path}...')
    #bidaf_model = util.load_model(bidaf_model, args.load_path, args.gpu_ids, return_step=False) # don't need step since we aren't training
    #bidaf_model = bidaf_model.to(device)
    bidaf_model.eval()                  # we eval only (vs train)

    # Setup the Paraphraser model
    #ema = util.EMA(bidaf_model, args.ema_decay)

    # Get saver
    # saver = util.CheckpointSaver(args.save_dir,
    #                              max_checkpoints=args.max_checkpoints,
    #                              metric_name=args.metric_name,
    #                              maximize_metric=args.maximize_metric,
    #                              log=log)

    # Get optimizer and scheduler
    # optimizer = optim.Adadelta(model.parameters(), args.lr,
    #                            weight_decay=args.l2_wd)
    # scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # Get data loader

    a = np.array([5, 1, 3, 4],
                 [4, 2, 1])
    c_idx = torch.from_numpy(a).long()
    #c_mask = torch.zeros_like(c_idx) != c_idx
    embeddings = bidaf_model.emb(c_idx)
    #c_len = c_mask.sum(-1)
    #print(f'c_len = {c_len}')
    #pp(c_idx)
    pp(embeddings)



if __name__ == '__main__':
    main(get_train_args())
