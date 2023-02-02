# Perceiver AR: Piano Continuations
**Author**: Tyler Vergho

**Class**: COSC 89, Music and AI, Winter 2023

## Description

Perceiver AR (original repository [here](https://github.com/google-research/perceiver-ar)) is a transformer model based on cross-attention that has been trained on long-range input contexts of up to 65,536 tokens. The model is general-purpose – it can be applied to books, images, and musical performances to generate outputs with extensive coherence and structure. This project mirrors the work done in the [original paper](https://arxiv.org/abs/2202.07765) by training the model – tuned with various hyperparameter configurations – on an open-source dataset of piano performances.

To use the model, an input MIDI file is then supplied as a primer sequence. The model then takes this input and attempts to generate a continuation in the style of the original song. The output is a MIDI file that can be played back in a MIDI player or converted to audio. 

The models are obviously too large to upload to GitHub or Canvas. The checkpoint files can be accessed through [this Google Drive link](https://drive.google.com/drive/folders/19weYvxuSZro-UoMWZ235VFbnxypz6gvw?usp=share_link).

## Usage
- Download the Maestro v3 dataset from [here](https://magenta.tensorflow.org/datasets/maestro#v300) and extract it to the `/maestro-v3.0.0"` directory. Alternatively, you may use a different dataset, but you will need to modify the `prep.py` file to load the new dataset.
- `pip install pretty_midi perceiver_ar_pytorch accelerate tqdm torch==1.11`
  - Install [Torch XLA](https://github.com/pytorch/xla) if needed for TPU compatibility.
- Run `prep.py` to preprocess the dataset and save the encoded MIDIs as `.pickle` files for later use by the model.
- Run `train.py` to train the model. Adjust hyperparameters using the constants at the top and in the `main` function. After training, the model checkpoint will be saved to the `/ckpt` directory.
  - You may then need to run the `cpu_convert.py` script to convert the model checkpoint to a CPU-compatible format. This is only necessary if you are running on a TPU.
- Run `generate.py` to generate samples from the model. Adjust hyperparameters using the constants at the top and in the `main` function. Also make sure to select a priming file. The generated samples will be saved to the `/output` directory.
- `train.py` and `generate.py` are designed to be used with the `accelerate` package, so they should be run with `accelerate launch` instead of `python`. For more information on `accelerate`, see [here](https://huggingface.co/docs/accelerate/).
- If running on a Google Cloud TPU, be sure to update your environment configuration so that the TPU is recognized by PyTorch. See [here](https://cloud.google.com/tpu/docs/pytorch-xla-ug-tpu-vm) for instructions.

When generating new samples, the parameters that seem to have the most impact on the output are the `cross_attn_seq_len` (controlling the context size of the cross-attention layer), `num_prime` (which controls the length of the primer sequence – larger values will result in a shorter generated output), and `temperature` (which controls the randomness of the output – higher values will result in more random output). These parameters can be adjusted in the `generate.py` file.

Note that inference from a pre-trained model demands far less compute power than training. Most of the outputs were generated through [Kaggle notebooks](https://www.kaggle.com/docs/notebooks) (which provide 30 hours of free GPU usage/week) on two NVIDIA T4 GPUs. Depending on parameters, the 8192 model takes around 45 minutes to generate a full-length sample on that setup.

## Organization

For the best, final versions of each model's output, refer to the files within the `/samples/successes` directory. Individual samples from different iterations of each model can be found in the subfolders of the `/samples` directory. Where possible, the original "primer" MIDI file is included in the same folder as the generated output. MIDI samples were converted to `.mp3` files using VLC Media Player. A couple common primers used across models are included in the `/samples/primers` directory.

## Models

4 different models were trained based on the Perceiver AR architecture:
- v1: 2048 context length, 512 dimensions, 8 attention heads, 8 layers
- v2: 4096 context length, 512 dimensions, 8 attention heads, 8 layers
- v3: 8192 context length, 1024 dimensions, 8 attention heads, 8 layers
- v4: 16384 context length, 512 dimensions, 4 attention heads, 8 layers

A fifth model with significantly increased dimensionality (d=2048 instead of d=1024 or d=512) and context length of 4096 was in the process of being trained as well, but some technical issues forced a restart before the deadline. When complete, the Drive folder and GitHub repository will be updated with samples and the model checkpoint.

The dataset used is [MAESTRO v3](https://magenta.tensorflow.org/datasets/maestro#v300) (MIDI and Audio Edited for Synchronous TRacks and Organization), which is the same dataset referenced in the music training section in the original Perceiver AR paper. It consists of around 200 hours of piano performances stored in the MIDI format.

As expected, training the models took successively more time, memory, and compute power as the context length, layers, and dimensions increased. All models (except for the 2048 version) were trained on [Google Cloud TPU](https://cloud.google.com/tpu) v2-8 and v3-8 VMs. These were provided free of charge through the [TPU Research Cloud](https://sites.research.google/trc/about/) program (and would've been a sizeable compute bill otherwise!) The largest model (v4) took about two days to train on a single TPUv3-8 VM. Other models took between 12-24 hours, depending on the context length and dimensionality.

The 16384 model attained a best cross-entropy loss value of 1.52 after 38 epochs of training. The 8192 model attained a best loss of 2.05.

The most obvious difference between the models is the length of the generated output. The 2048 model generates up to a 1 minute output depending on the length and tempo of the primer used, while the 16384 model could generate up to 5-10 minutes of output (the provided samples were cut off early due to time/memory constraints). Some additional quality and continuity improvements can be observed in the 8192 and 16384 models, but the differences are not as drastic as the differences in output length.

## Challenges

Some challenges faced during the training and generation process included:
- **Pytorch/Research Code**: The original codebase was written in Jax, the learning curve for which I eventually concluded was too steep for me to take on for this assignment. I eventually found an [implementation](https://github.com/lucidrains/perceiver-ar-pytorch) of the model in Pytorch, but it was still a bit difficult to understand and work with (especially as I had minimal experience with Pytorch or deep learning models previously). I had to make some modifications to the code for it to work with the Google Cloud TPU VMs (as opposed to the original GPUs), and successfully train and generate samples from the models. The [Perceiver Music Generation](https://github.com/feizc/Perceiver-Music-Generation) repository was extremely helpful in providing an example of how to train the model to generate music output, as well as providing code for specific tasks like encoding/decoding MIDI files and decoding the model's output.
- **Memory**: Training with a context size beyond 2048, even with a batch size of 1, demanded GPU memory beyond the capacity of a free Google Colab instance. The v3-8 version of TPUs from Google provide 8 cores with 16 GB of memory per core, which was enough to train the models with the above parameters with this setup. However, attempting to exceed these limits – for instance, by using 8 attention heads or a depth of 12 on the 16384 model – quickly led to out of memory errors. All the models were trained using a distributed strategy across multiple TPUs using the Huggingface [accelerate](https://huggingface.co/docs/accelerate/package_reference/accelerator) package. 
- **Repetitive Output**: Several samples demonstrate that the model is prone to generating repetitive output (where the same note tends to repeat over and over for the duration of the song). This is a common problem with generative models, and is not unique to the Perceiver AR architecture. More experimentation with the model's hyperparameters and training data could help to mitigate this issue.
- **Campus WiFi**: It'd occasionally cut out, forcing restart of training/generation in a couple cases when scripts were being run through SSH.

## Acknowledgements
- [Perceiver AR Pytorch](https://github.com/lucidrains/perceiver-ar-pytorch)
- [Perceiver Music Generation](https://github.com/feizc/Perceiver-Music-Generation)
- Perceiver AR [paper](https://arxiv.org/abs/2202.07765) and [code](https://github.com/google-research/perceiver-ar)
- [Google Cloud TPU](https://cloud.google.com/tpu) and the [TPU Research Cloud](https://sites.research.google/trc/about/) for compute power and storage.
- [MIDI processor](https://github.com/jason9693/midi-neural-processor) – used for encoding/decoding and loading/saving MIDI files.
- [Kaggle](https://www.kaggle.com/docs/notebooks) and [Google Colab](https://colab.research.google.com/)