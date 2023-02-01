# Most of this code is overlap with the MIDI processing code in prep.py, but Python was giving relative import errors so I just copied everything here. 
# Additional code was adapted from https://github.com/feizc/Perceiver-Music-Generation/blob/main/generate.py
import torch.nn.functional as F
import os 
import torch 
import torch.nn as nn
import pretty_midi 
from torch.optim.lr_scheduler import LambdaLR 
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam 
import argparse
from tqdm import tqdm 
import numpy as np
import random
import math
import pickle

from perceiver_ar_pytorch import PerceiverAR 
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device

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

TOKEN_END               = RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VEL + RANGE_TIME_SHIFT
TOKEN_PAD               = TOKEN_END + 1

VOCAB_SIZE              = TOKEN_PAD + 1

TORCH_FLOAT             = torch.float32
TORCH_INT               = torch.int32

TORCH_LABEL_TYPE        = torch.long

PREPEND_ZEROS_WIDTH     = 4 

class SustainAdapter:
    def __init__(self, time, type):
        self.start =  time
        self.type = type


class SustainDownManager:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.managed_notes = []
        self._note_dict = {} # key: pitch, value: note.start

    def add_managed_note(self, note: pretty_midi.Note):
        self.managed_notes.append(note)

    def transposition_notes(self):
        for note in reversed(self.managed_notes):
            try:
                note.end = self._note_dict[note.pitch]
            except KeyError:
                note.end = max(self.end, note.end)
            self._note_dict[note.pitch] = note.start


# Divided note by note_on, note_off
class SplitNote:
    def __init__(self, type, time, value, velocity):
        ## type: note_on, note_off
        self.type = type
        self.time = time
        self.velocity = velocity
        self.value = value

    def __repr__(self):
        return '<[SNote] time: {} type: {}, value: {}, velocity: {}>'\
            .format(self.time, self.type, self.value, self.velocity)


class Event:
    def __init__(self, event_type, value):
        self.type = event_type
        self.value = value

    def __repr__(self):
        return '<Event type: {}, value: {}>'.format(self.type, self.value)

    def to_int(self):
        return START_IDX[self.type] + self.value

    @staticmethod
    def from_int(int_value):
        info = Event._type_check(int_value)
        return Event(info['type'], info['value'])

    @staticmethod
    def _type_check(int_value):
        range_note_on = range(0, RANGE_NOTE_ON)
        range_note_off = range(RANGE_NOTE_ON, RANGE_NOTE_ON+RANGE_NOTE_OFF)
        range_time_shift = range(RANGE_NOTE_ON+RANGE_NOTE_OFF,RANGE_NOTE_ON+RANGE_NOTE_OFF+RANGE_TIME_SHIFT)

        valid_value = int_value

        if int_value in range_note_on:
            return {'type': 'note_on', 'value': valid_value}
        elif int_value in range_note_off:
            valid_value -= RANGE_NOTE_ON
            return {'type': 'note_off', 'value': valid_value}
        elif int_value in range_time_shift:
            valid_value -= (RANGE_NOTE_ON + RANGE_NOTE_OFF)
            return {'type': 'time_shift', 'value': valid_value}
        else:
            valid_value -= (RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT)
            return {'type': 'velocity', 'value': valid_value}


def _divide_note(notes):
    result_array = []
    notes.sort(key=lambda x: x.start)

    for note in notes:
        on = SplitNote('note_on', note.start, note.pitch, note.velocity)
        off = SplitNote('note_off', note.end, note.pitch, None)
        result_array += [on, off]
    return result_array


def _merge_note(snote_sequence):
    note_on_dict = {}
    result_array = []

    for snote in snote_sequence:
        # print(note_on_dict)
        if snote.type == 'note_on':
            note_on_dict[snote.value] = snote
        elif snote.type == 'note_off':
            try:
                on = note_on_dict[snote.value]
                off = snote
                if off.time - on.time == 0:
                    continue
                result = pretty_midi.Note(on.velocity, snote.value, on.time, off.time)
                result_array.append(result)
            except:
                print('info removed pitch: {}'.format(snote.value))
    return result_array


def _snote2events(snote: SplitNote, prev_vel: int):
    result = []
    if snote.velocity is not None:
        modified_velocity = snote.velocity // 4
        if prev_vel != modified_velocity:
            result.append(Event(event_type='velocity', value=modified_velocity))
    result.append(Event(event_type=snote.type, value=snote.value))
    return result


def _event_seq2snote_seq(event_sequence):
    timeline = 0
    velocity = 0
    snote_seq = []

    for event in event_sequence:
        if event.type == 'time_shift':
            timeline += ((event.value+1) / 100)
        if event.type == 'velocity':
            velocity = event.value * 4
        else:
            snote = SplitNote(event.type, timeline, event.value, velocity)
            snote_seq.append(snote)
    return snote_seq


def _make_time_sift_events(prev_time, post_time):
    time_interval = int(round((post_time - prev_time) * 100))
    results = []
    while time_interval >= RANGE_TIME_SHIFT:
        results.append(Event(event_type='time_shift', value=RANGE_TIME_SHIFT-1))
        time_interval -= RANGE_TIME_SHIFT
    if time_interval == 0:
        return results
    else:
        return results + [Event(event_type='time_shift', value=time_interval-1)]


def _control_preprocess(ctrl_changes):
    sustains = []

    manager = None
    for ctrl in ctrl_changes:
        if ctrl.value >= 64 and manager is None:
            # sustain down
            manager = SustainDownManager(start=ctrl.time, end=None)
        elif ctrl.value < 64 and manager is not None:
            # sustain up
            manager.end = ctrl.time
            sustains.append(manager)
            manager = None
        elif ctrl.value < 64 and len(sustains) > 0:
            sustains[-1].end = ctrl.time
    return sustains


def _note_preprocess(susteins, notes):
    note_stream = []

    for sustain in susteins:
        for note_idx, note in enumerate(notes):
            if note.start < sustain.start:
                note_stream.append(note)
            elif note.start > sustain.end:
                notes = notes[note_idx:]
                sustain.transposition_notes()
                break
            else:
                sustain.add_managed_note(note)

    for sustain in susteins:
        note_stream += sustain.managed_notes

    note_stream.sort(key= lambda x: x.start)
    return note_stream

def encode_midi(file_path):
    events = []
    notes = []
    mid = pretty_midi.PrettyMIDI(midi_file=file_path)

    for inst in mid.instruments:
        inst_notes = inst.notes
        # ctrl.number is the number of sustain control. If you want to know abour the number type of control,
        # see https://www.midi.org/specifications-old/item/table-3-control-change-messages-data-bytes-2
        ctrls = _control_preprocess([ctrl for ctrl in inst.control_changes if ctrl.number == 64])
        notes += _note_preprocess(ctrls, inst_notes)

    dnotes = _divide_note(notes)

    # print(dnotes)
    dnotes.sort(key=lambda x: x.time)
    # print('sorted:')
    # print(dnotes)
    cur_time = 0
    cur_vel = 0
    for snote in dnotes:
        events += _make_time_sift_events(prev_time=cur_time, post_time=snote.time)
        events += _snote2events(snote=snote, prev_vel=cur_vel)
        # events += _make_time_sift_events(prev_time=cur_time, post_time=snote.time)

        cur_time = snote.time
        cur_vel = snote.velocity

    return [e.to_int() for e in events]


def decode_midi(idx_array, file_path=None):
    event_sequence = [Event.from_int(idx) for idx in idx_array]
    # print(event_sequence)
    snote_seq = _event_seq2snote_seq(event_sequence)
    note_seq = _merge_note(snote_seq)
    note_seq.sort(key=lambda x:x.start)

    mid = pretty_midi.PrettyMIDI()
    # if want to change instument, see https://www.midi.org/specifications/item/gm-level-1-sound-set
    instument = pretty_midi.Instrument(1, False, "Developed By Yang-Kichang")
    instument.notes = note_seq

    mid.instruments.append(instument)
    if file_path is not None:
        mid.write(file_path)
    return mid

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

def greedy_decode(condi, model, max_sequence): 
    ys = [] 
    for i in range(max_sequence): 
        if i > 0: 
            predict = torch.tensor(ys).long().unsqueeze(0).to(device)
            input = torch.cat((condi, predict), dim=-1).to(device)
        else: 
            input = condi
        out = model(input)
        logits = out[0][-1, :].cpu().data.numpy() 
        next_token = np.argsort(logits)[-1] 
        ys.append(next_token) 
    return ys 

def sample_sequence(condi, model, max_sequence, temperature=0.7, top_k=0, top_p=0.9): 
    ys = [] 
    for i in range(max_sequence): 
        if i > 0: 
            predict = torch.tensor(ys).long().unsqueeze(0).to(device)
            input = torch.cat((condi, predict), dim=-1).to(device)
        else: 
            input = condi 
        out = model(input)
        logits = out[0][-1, :] / temperature 
        logits = top_filtering(logits, top_k=top_k, top_p=top_p) 
        probs = F.softmax(logits, dim=-1)
        
        next_token = torch.multinomial(probs, 1).item() 
        max_iter = 0 
        while next_token == TOKEN_END or next_token == TOKEN_PAD: 
            next_token = torch.multinomial(probs, 1).item() 
            max_iter += 1 
            if max_iter > 100: 
                next_token = TOKEN_END - 1 
        ys.append(next_token)
        
    return ys 



def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

def generate():
    output_dir = "./output"
    primer_file = os.path.join('input.midi') # Change this to your midi priming file
    n_workers = 1
    batch_size = 1
    max_sequence = 16384
    d_model = 512
    num_prime = 1024
    temperature = 0.7
    
    os.makedirs(output_dir, exist_ok=True)
    
    model = PerceiverAR(
        num_tokens = VOCAB_SIZE, 
        dim = d_model, 
        depth = 8, 
        dim_head = 64, 
        heads = 4, 
        max_seq_len = max_sequence, 
        cross_attn_seq_len = int((num_prime)/2), 
        cross_attn_dropout = 0.7,
    )
    model.load_state_dict(torch.load(os.path.join('latest.pth'))['state_dict']) # Adjust this to your model path
    model = model.to(device) 

    model.eval()
    
    raw_mid = encode_midi(primer_file)
    primer, _  = process_midi(raw_mid, num_prime, random_seq=True)
    prime_array = primer[:num_prime]
    primer = torch.tensor(primer, dtype=TORCH_LABEL_TYPE, device=device)
    f_path = os.path.join(output_dir, "primer.mid")
    decode_midi(primer[:num_prime].cpu().numpy(), file_path=f_path)
    
    o_file = os.path.join(output_dir, 'pickle/primer.pickle')
    os.makedirs(os.path.join(output_dir, 'pickle'), exist_ok=True)
    o_stream = open(o_file, "wb")
    pickle.dump(prime_array, o_stream)
    o_stream.close()
    
    test_dataset = EPianoDataset(os.path.join(output_dir, 'pickle'), max_sequence, False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=n_workers)
    model, test_loader = accelerator.prepare(model, test_loader)

    with torch.no_grad(): 
        with tqdm(enumerate(test_loader), total=len(test_loader)) as t: 
            for batch_num, batch in t: 
                x = batch[0].to(device) 
                tgt = batch[1].to(device) 
                condi = x[:, :num_prime] 
                
                predict = sample_sequence(condi, model, max_sequence - num_prime - 1, temperature=temperature) 
                
                predict = np.array(predict) 
                midi_name = 'sample_' + str(1) + '.mid' 
                midi_path = os.path.join(output_dir, midi_name)
                decode_midi(predict, file_path=midi_path) 
                break

generate()