# Helper script to convert the model to cpu, to run on Kaggle/Colab/other platforms
# Needed to make models trained on TPU compatible with other platforms
from perceiver_ar_pytorch import PerceiverAR 
import os 
import torch 
import torch_xla.core.xla_model as xm

device = xm.xla_device()
torch.device(device)

RANGE_NOTE_ON = 128
RANGE_NOTE_OFF = 128
RANGE_VEL = 32
RANGE_TIME_SHIFT = 100
TOKEN_END               = RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VEL + RANGE_TIME_SHIFT
TOKEN_PAD               = TOKEN_END + 1
VOCAB_SIZE              = TOKEN_PAD + 1

max_sequence = 4096
dimensions = 512
heads = 8

model = PerceiverAR(
        num_tokens = VOCAB_SIZE, 
        dim = dimensions, 
        depth = 8, 
        dim_head = 64, 
        heads = heads, 
        max_seq_len = max_sequence, 
        cross_attn_seq_len = 1536,
        cross_attn_dropout = 0.7,
    )
model.load_state_dict(torch.load(os.path.join('latest-ckpt.pth'))['state_dict'])
model = model.to('cpu')
torch.save({'state_dict': model.state_dict()}, os.path.join('latest-cpu.pth'))
