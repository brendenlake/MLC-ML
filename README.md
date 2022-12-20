# Applying Behaviorally-Informed Meta-Learning (BIML) to machine learning benchmarks

BIML is a meta-learning approach for guiding neural networks to human-like systematic generalization and inductive biases. This code shows how to train and evaluate a modified sequence-to-sequence (seq2seq) transformer for memory-based meta-learning. This repository shows how to apply BIML-Scale to the [SCAN](https://github.com/brendenlake/SCAN) and [COGS](https://github.com/najoungkim/COGS) machine learning benchmarks. The more basic architecture applied to modeling human behavior is available [here](https://github.com/brendenlake/BIML-sysgen).

This code accompanies the following submitted paper.
- Lake, B. M. and Baroni, M. (submitted). Human-like systematic generalization through a meta-learning neural network.   

You can email brenden AT nyu.edu if you would like a copy.

## Credits
This repo borrows from the excellent [PyTorch seq2seq tutorial](https://pytorch.org/tutorials/beginner/translation_transformer.html).

## Requirements
Python 3 with the following packages:
torch (PyTorch), sklearn (scikit-learn), numpy, matplotlib

## Downloading pre-trained models
To get pre-trained models, you should download the following [zip file](https://cims.nyu.edu/~brenden/supplemental/BIML-large-files/BIML_ml_models.zip). Please extract `BIML_ml_models.zip` such that `out_models_scan` and `out_models_cogs` are sub-directories of the main repo and contain the model files `net_*.pt.`

## Evaluating models
Models are evaluated via their best responses to the test commands. Here we find the best response from the pre-trained BIML model using greedy decoding:
```python
python eval.py  --max --episode_type X --fn_out_model X.pt --verbose
```

The full set of evaluation arguments can be viewed with when typing `python eval.py -h`:
```
optional arguments:
  -h, --help            show this help message and exit
  --fn_out_model FN_OUT_MODEL
                        *REQUIRED*. Filename for loading the model
  --dir_model DIR_MODEL
                        Directory for loading the model file
  --max_length_eval MAX_LENGTH_EVAL
                        Maximum generated sequence length (must be at least 50
                        for SCAN and 400 for COGS)
  --batch_size BATCH_SIZE
                        Maximum generated sequence length
  --episode_type EPISODE_TYPE
                        What type of episodes do we want? See datasets.py for
                        options
  --dashboard           Showing loss curves during training.
  --ll                  Evaluate log-likelihood of validation (val) set
  --max                 Find best outputs for val commands (greedy decoding)
  --debug
  --verbose             Inspect outputs in more detail
```

## Episode types
Please see `datasets.py` for the full set of options. Here are a few key episode types that can be set via `--episode_type`:

For evaluating on SCAN:
- `simple_actions` : For meta-training and evaluating on SCAN Simple (IID) split.
- `addprim_jump_actions` : For meta-training and evaluating on SCAN Add jump split.
- `around_right_actions` : For meta-training and evaluating on SCAN Around right split.
- `opposite_right_actions` : For meta-training and evaluating on SCAN Opposite right split.
- `length_actions` : For meta-training and evaluating on SCAN Length split.

For BIML on COGS:
- `cogs_train_targeted` : Meta-training for COGS on all splits.
- `cogs_iid` : For evaluating on COGS Simple (IID) split.
- `cogs_gen_lex` : For evaluating on systematic lexical generalization split.
- `cogs_gen_struct` : for evaluating on systematic structural generalization split.

## Training models from scratch
Here are example of how to optimize BIML-Scale on SCAN and COGS. For training and evaluating on the SCAN Add jump split, 
```python
python train.py --episode_type addprim_jump_actions --fn_out_model net-BIML-scan-add-jump.pt
```
which after training will produce a file `out_models/net-BIML-scan-add-jump.pt`. For training and evaluating on the other SCAN splits, replace the `episode_type` with the options above.

To optimize for all COGS splits,
```python
python train.py --episode_type cogs_train_targeted --batch_size 40 --nepochs 300 --fn_out_model net-BIML-cogs.pt
```
which after training will produce a file `out_models/net-BIML-cogs.pt`.


The full set of training arguments can be viewed with `python train.py -h`:
```
optional arguments:
  -h, --help            show this help message and exit
  --fn_out_model FN_OUT_MODEL
                        *REQUIRED* Filename for saving model checkpoints.
                        Typically ends in .pt
  --dir_model DIR_MODEL
                        Directory for saving model files
  --episode_type EPISODE_TYPE
                        What type of episodes do we want? See datasets.py for
                        options
  --batch_size BATCH_SIZE
                        number of episodes per batch
  --batch_hold_update BATCH_HOLD_UPDATE
                        update the weights after this many batches (default=1)
  --nepochs NEPOCHS     number of training epochs
  --lr LR               learning rate
  --lr_end_factor LR_END_FACTOR
                        factor X for decrease learning rate linearly from
                        1.0*lr to X*lr across training
  --no_lr_warmup        Turn off learning rate warm up (by default, we use 1
                        epoch of warm up)
  --emb_size EMB_SIZE   size of embedding
  --nlayers_encoder NLAYERS_ENCODER
                        number of layers for encoder
  --nlayers_decoder NLAYERS_DECODER
                        number of layers for decoder
  --ff_mult FF_MULT     multiplier for size of the fully-connected layer in
                        transformer
  --dropout DROPOUT     dropout applied to embeddings and transformer
  --act ACT             activation function in the fully-connected layer of
                        the transformer (relu or gelu)
  --save_best           Save the "best model" according to validation loss.
  --save_best_skip SAVE_BEST_SKIP
                        Do not bother saving the "best model" for this
                        fraction of early training
  --resume              Resume training from a previous checkpoint
```
Note that the `save_best` options were not used in any benchmark experiment and should not be used in these use cases. SCAN does not have a validation set, and early stopping on the COGS validation set leads to worse performance                       