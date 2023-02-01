import os
import pickle
import random
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.optim import DistributedOptimizer
from torch.utils.data import Dataset
import math 
import torch 
import torch_xla
import torch_xla.core.xla_model as xm
from torch.optim.lr_scheduler import LambdaLR 
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam 
import argparse
from tqdm import tqdm 
import torch.multiprocessing as mp
from perceiver_ar_pytorch import PerceiverAR 
from accelerate import Accelerator

accelerator = Accelerator()

SEPERATOR               = "========================="

RANGE_NOTE_ON = 128
RANGE_NOTE_OFF = 128
RANGE_VEL = 32
RANGE_TIME_SHIFT = 100


START_IDX = {
    'note_on': 0,
    'note_off': RANGE_NOTE_ON,
    'time_shift': RANGE_NOTE_ON + RANGE_NOTE_OFF,
    'velocity': RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT
}

# Taken from the paper
ADAM_BETA_1             = 0.1
ADAM_BETA_2             = 0.999
ADAM_EPSILON            = 1e-8

LR_DEFAULT_START        = 1.0
SCHEDULER_WARMUP_STEPS  = 4000
# LABEL_SMOOTHING_E       = 0.1

# DROPOUT_P               = 0.1

TOKEN_END               = RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VEL + RANGE_TIME_SHIFT
TOKEN_PAD               = TOKEN_END + 1

VOCAB_SIZE              = TOKEN_PAD + 1

TORCH_FLOAT             = torch.float32
TORCH_INT               = torch.int32

TORCH_LABEL_TYPE        = torch.long

PREPEND_ZEROS_WIDTH     = 4 

device = accelerator.device

SEQUENCE_START = 0

class EPianoDataset(Dataset):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Pytorch Dataset for the Maestro e-piano dataset (https://magenta.tensorflow.org/datasets/maestro).
    Recommended to use with Dataloader (https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
    Uses all files found in the given root directory of pre-processed (preprocess_midi.py)
    Maestro midi files.
    ----------
    """

    def __init__(self, root, max_seq=2048, random_seq=True):
        self.root       = root
        self.max_seq    = max_seq
        self.random_seq = random_seq

        fs = [os.path.join(root, f) for f in os.listdir(self.root)]
        self.data_files = [f for f in fs if os.path.isfile(f)]

    # __len__
    def __len__(self):
        """
        ----------
        Author: Damon Gwinn
        ----------
        How many data files exist in the given directory
        ----------
        """

        return len(self.data_files)

    # __getitem__
    def __getitem__(self, idx):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Gets the indexed midi batch. Gets random sequence or from start depending on random_seq.
        Returns the input and the target.
        ----------
        """

        # All data on cpu to allow for the Dataloader to multithread
        i_stream    = open(self.data_files[idx], "rb")
        # return pickle.load(i_stream), None
        raw_mid     = torch.tensor(pickle.load(i_stream), dtype=TORCH_LABEL_TYPE, device=torch.device("cpu"))
        i_stream.close()

        x, tgt = process_midi(raw_mid, self.max_seq, self.random_seq)

        return x, tgt



def process_midi(raw_mid, max_seq, random_seq):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Takes in pre-processed raw midi and returns the input and target. Can use a random sequence or
    go from the start based on random_seq.
    ----------
    """

    x   = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=torch.device("cpu"))
    tgt = torch.full((max_seq, ), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=torch.device("cpu"))

    raw_len     = len(raw_mid)
    full_seq    = max_seq + 1 # Performing seq2seq

    if(raw_len == 0):
        return x, tgt

    if(raw_len < full_seq):
        x[:raw_len]         = raw_mid
        tgt[:raw_len-1]     = raw_mid[1:]
        tgt[raw_len-1]      = TOKEN_END
    else:
        # Randomly selecting a range
        if(random_seq):
            end_range = raw_len - full_seq
            start = random.randint(SEQUENCE_START, end_range)

        # Always taking from the start to as far as we can
        else:
            start = SEQUENCE_START

        end = start + full_seq

        data = raw_mid[start:end]

        x = data[:max_seq]
        tgt = data[1:full_seq]


    # print("x:",x)
    # print("tgt:",tgt)

    return x, tgt 


# create_epiano_datasets
def create_epiano_datasets(dataset_root, max_seq, random_seq=True):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Creates train, evaluation, and test EPianoDataset objects for a pre-processed (preprocess_midi.py)
    root containing train, val, and test folders.
    ----------
    """

    train_root = os.path.join(dataset_root, "train")
    val_root = os.path.join(dataset_root, "val")
    test_root = os.path.join(dataset_root, "test")

    train_dataset = EPianoDataset(train_root, max_seq, random_seq)
    val_dataset = EPianoDataset(val_root, max_seq, random_seq)
    test_dataset = EPianoDataset(test_root, max_seq, random_seq)

    return train_dataset, val_dataset, test_dataset 

def train(cur_epoch, model, data_loader, opt): 
    model.train() 
    sum_loss = .0 
    sum_acc = .0 
    with tqdm(enumerate(data_loader), total=len(data_loader)) as t: 
        for batch_num, batch in t: 
            opt.zero_grad()
            x   = batch[0].to(device)
            tgt = batch[1].to(device) 
            out = model(x, labels=tgt) 
            accelerator.backward(out)
            opt.step() 
            sum_loss += out.item()
            sum_acc += compute_accuracy(out, tgt)
            t.set_description('Epoch %i' % cur_epoch)
            t.set_postfix(loss=sum_loss / (batch_num+1), acc=sum_acc/(batch_num+1))  
            
def eval(model, data_loader): 
    model.eval() 
    sum_loss = .0 
    sum_acc = .0 
    with torch.no_grad(): 
        with tqdm(enumerate(data_loader), total=len(data_loader)) as t: 
            for batch_num, batch in t: 
                x = batch[0].to(device) 
                tgt = batch[1].to(device) 
                out = model(x, labels=tgt) 
                loss = out.item()
                sum_loss += loss
                sum_acc += compute_accuracy(out, tgt)
                t.set_description('Evaluation')
                t.set_postfix(loss=sum_loss / (batch_num+1), acc=sum_acc/(batch_num+1))
    return sum_acc/(batch_num+1) 

def main(): 
    # Adjust parameters as needed
    data_dir = "./data"
    ckpt_dir = "./ckpt"
    lr = 2e-4
    n_workers = 1
    batch_size = 1
    epochs = 100
    max_sequence = 4096
    d_model = 2048

    os.makedirs(ckpt_dir, exist_ok=True) 

    ##### Datasets #####
    train_dataset, val_dataset, test_dataset = create_epiano_datasets(data_dir, max_sequence) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=n_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=n_workers) 

    ##### Model #####
    model = PerceiverAR(
        num_tokens = VOCAB_SIZE, 
        dim = d_model, 
        depth = 8,
        dim_head = 64, 
        heads = 16, 
        max_seq_len = max_sequence, 
        cross_attn_seq_len = 1024, 
        cross_attn_dropout = 0.7
    )

    opt = Adam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)
    model, opt, train_loader, val_loader = accelerator.prepare(model, opt, train_loader, val_loader)

    best_loss = float('-inf')
    for epoch in range(epochs): 
        train(epoch, model, train_loader, opt)
        acc = eval(model, val_loader)
        if acc > best_acc: 
            best_loss = loss
            print(f"Saving epoch {epoch} with loss {loss}")
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer': opt.state_dict(),
            }, os.path.join(ckpt_dir, 'latest.pth'))

main()