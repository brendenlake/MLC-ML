import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import glob
import random
import numpy as np
import re
import pickle
from copy import deepcopy, copy
from train_lib import seed_all, display_input_output, list_remap
import matplotlib.pyplot as plt
from sklearn import utils
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generating episodes for meta-training and evaluation. 
#   Includes SCAN and COGS
#
# We also include code for training vanilla seq2seq models on SCAN and COGS.

my_ns_default = 10 # default number of support items

# Special symbols
SOS_token = "<SOS>" # start of sentence
EOS_token = "<EOS>" # end of sentence
PAD_token = "<PAD>" # padding symbol
IO_SEP = '->' # separator '->' between input/outputs in support examples
ITEM_token  = '|' # separator '|' between support examples in input sequence

# Default input and output symbols (for SCAN)
input_symbols_list_default = ['right', 'turn', 'look', 'twice', 'around', 'run', 'thrice', 'walk', 'after', 'left', 'and', 'opposite', 'jump']
output_symbols_list_default = ['I_JUMP', 'I_WALK', 'I_RUN', 'I_TURN_LEFT', 'I_LOOK', 'I_TURN_RIGHT']

# --------------------------------------------
# Datasets for main experiments
# --------------------------------------------
def get_dataset(episode_type):
    # BIML SCAN
    if episode_type == 'addprim_jump_actions':
        D_train = DataSCAN('addprim_jump','train','actions')
        D_val = DataSCAN('addprim_jump','test','NONE')
    elif episode_type == 'addprim_left_actions':
        D_train = DataSCAN('addprim_left','train','actions')
        D_val = DataSCAN('addprim_left','test','NONE')
    elif episode_type == 'around_right_actions':
        D_train = DataSCAN('around_right','train','actions')
        D_val = DataSCAN('around_right','test','NONE')
    elif episode_type == 'simple_actions':
        D_train = DataSCAN('simple','train','actions')
        D_val = DataSCAN('simple','test','NONE')
    elif episode_type == 'opposite_right_actions':
        D_train = DataSCAN('opposite_right','train','actions')
        D_val = DataSCAN('opposite_right','test','NONE')
    elif episode_type == 'length_actions':
        D_train = DataSCAN('length','train','actions')
        D_val = DataSCAN('length','test','NONE')
    # BIML COGS
    elif episode_type == 'cogs_train_targeted': # min ns=8 # COGS meta-training
        D_train = DataCOGS('train','targeted',ns=8)
        D_val = DataCOGS('dev_tiny','NONE',ns=8,inc_support_in_query=False)
    elif episode_type == 'cogs_gen_lex': # min ns=8
        D_train = []
        D_val = DataCOGS('gen_lexical','NONE',ns=8,inc_support_in_query=False)
    elif episode_type == 'cogs_iid': # min ns=8
        D_train = []
        D_val = DataCOGS('test_iid','NONE',ns=8,inc_support_in_query=False) # COGS Simple
    elif episode_type == 'cogs_gen_struct': # min ns=18
        D_train = []
        D_val = DataCOGS('gen_structural','NONE',ns=18,inc_support_in_query=False)
    # Vanilla SCAN (Basic seq2seq)
    elif episode_type == 'addprim_jump_vanilla':
        D_train = VanillaDataSCAN('addprim_jump','train')
        D_val = VanillaDataSCAN('addprim_jump','test')
    elif episode_type == 'addprim_left_vanilla':
        D_train = VanillaDataSCAN('addprim_left','train')
        D_val = VanillaDataSCAN('addprim_left','test')
    elif episode_type == 'around_right_vanilla':
        D_train = VanillaDataSCAN('around_right','train')
        D_val = VanillaDataSCAN('around_right','test')
    elif episode_type == 'simple_vanilla':
        D_train = VanillaDataSCAN('simple','train')
        D_val = VanillaDataSCAN('simple','test')
    elif episode_type == 'opposite_right_vanilla':
        D_train = VanillaDataSCAN('opposite_right','train')
        D_val = VanillaDataSCAN('opposite_right','test')
    elif episode_type == 'length_vanilla':
        D_train = VanillaDataSCAN('length','train')
        D_val = VanillaDataSCAN('length','test')
    # Vanilla COGS (Basic seq2seq)
    elif episode_type == 'cogs_train_vanilla':
        D_train = VanillaCOGS('train')
        D_val = VanillaCOGS('dev_tiny')
    elif episode_type == 'cogs_iid_vanilla':
        D_train = []
        D_val = VanillaCOGS('test_iid')
    elif episode_type == 'cogs_gen_lex_vanilla':
        D_train = []
        D_val = VanillaCOGS('gen_lexical')
    elif episode_type == 'cogs_gen_struct_vanilla':
        D_train = []
        D_val = VanillaCOGS('gen_structural')
    else:
        assert False # invalid episode types
    return D_train, D_val

class Lang:
    #  Class for converting tokens strings to token index, and vice versa.
    #   Use separate class for input and output languages
    #
    def __init__(self, symbols):
        # symbols : list of all possible symbols besides special tokens SOS, EOS, and PAD
        n = len(symbols)
        assert(SOS_token not in symbols)
        assert(EOS_token not in symbols)
        assert(PAD_token not in symbols)
        assert(PAD_token != SOS_token)
        self.symbols = symbols # list of non-special symbols
        self.index2symbol = {n: SOS_token, n+1: EOS_token, n+2: PAD_token}
        self.symbol2index = {SOS_token : n, EOS_token : n+1, PAD_token : n+2}
        for idx,s in enumerate(symbols):
            self.index2symbol[idx] = s
            self.symbol2index[s] = idx
        self.n_symbols = len(self.index2symbol)
        self.PAD_idx = self.symbol2index[PAD_token]
        if ITEM_token in symbols: self.ITEM_idx = self.symbol2index[ITEM_token]
        self.PAD_token = PAD_token
        self.ITEM_token = ITEM_token
        assert(len(self.index2symbol)==len(self.symbol2index))

    def symbols_to_tensor(self, mylist, add_eos=True):
        # Convert a list of symbols to a tensor of indices (adding a EOS token at end)
        # 
        # Input
        #  mylist  : list of m symbols
        #  add_eos : true/false, if true add the EOS symbol at end
        #
        # Output
        #  output : [m or m+1 LongTensor] indices of each symbol (plus EOS if appropriate)
        mylist = copy(mylist)
        if add_eos: mylist.append(EOS_token)
        indices = [self.symbol2index[s] for s in mylist]
        output = torch.LongTensor(indices) # keep on CPU since this occurs inside Dataset getitem..
        return output

    def tensor_to_symbols(self, v):
        # Convert tensor of token index to token strings, breaking where we get a EOS token.
        #   The EOS token is not included at the end in the result string list.
        # 
        # Input
        #  v : python list of m indices, or 1D tensor
        #   
        # Output
        #  mylist : list of symbols (excluding EOS)
        if torch.is_tensor(v):
            assert v.dim()==1
            v = v.tolist()
        assert isinstance(v, list)
        mylist = []
        for x in v:
            s = self.index2symbol[x]
            if s == EOS_token:
                break
            mylist.append(s)
        return mylist

def readfile_scan(fn_in):
    # Read SCAN dataset from text file
    #
    # Input
    #  fn_in : filename to read
    #
    # Output
    #   Parsed version of the file, with struct
    #   
    fid = open(os.path.join(fn_in),'r')
    lines = fid.readlines()
    lines = [line.rstrip('\n') for line in lines]
    lines = [line for line in lines if line != '']
    assert(all([(' OUT: ' in line for line in lines)]))
    commands, outputs = parse_scan_commands(lines)
    fid.close()
    return commands, outputs

def parse_scan_commands(lines):
    # Parse lines from input files into command sequence and output sequence
    #
    # Input
    #  lines: [list of strings], each of format "IN: a b c OUT: d e f""
    lines = [l.strip() for l in lines]
    lines = [rmv_prefix(l,'IN: ') for l in lines]
    D = [l.split(' OUT: ') for l in lines]
    x = [d[0].split(' ') for d in D]
    y = [d[1].split(' ') for d in D]
    return x, y

def readfile_cogs(fn_in, make_output_upper=True):
    # Read COGS dataset from text file
    #
    # Input
    #  fn_in : filename to read
    #  make_output_upper (default=True) : convert the output tokens to uppercase, in order to 
    #   break correspondence with identical input tokens
    #
    # Output
    #   Parsed version of the file, with struct
    fid = open(os.path.join(fn_in),'r')
    lines = fid.readlines()
    lines = [line.rstrip('\n') for line in lines]
    lines = [line for line in lines if line != '']
    commands, outputs, type_labels = parse_cogs_commands(lines)
    if make_output_upper:
        for j in range(len(outputs)):
            outputs[j] = make_list_upper(outputs[j])
    fid.close()
    return commands, outputs, type_labels

def make_list_upper(mylist, special_symbols=['.', 'A', 'TV']):
    # Convert each token string in "mylist" to upper case.
    #  Note, for each of the "special_symbols" identified after upper transform,
    #  add an additional & marker to avoid repeat with input vocab 
    assert(isinstance(mylist,list))
    assert(isinstance(mylist[0],str))
    for j in range(len(mylist)):
        mylist[j] = mylist[j].upper()
        if mylist[j] in special_symbols:
            mylist[j] = '&' + mylist[j]
    return mylist

def parse_cogs_commands(lines):
    # Parse lines from input files into command sequence and output sequence
    #
    # Input
    #  lines: [list of strings], each of format "a b c \t d e f \t trial_type_labe""
    lines = [l.strip() for l in lines]
    D = [l.split('\t') for l in lines] # (input, output, type)
    x = [d[0].strip().split(' ') for d in D]
    y = [d[1].strip().split(' ') for d in D]
    type_labels = [d[2].strip() for d in D]
    return x, y, type_labels

def rmv_prefix(mystr,prefix):
    # If string begins with a prefix, remove it
    n = len(prefix)
    if mystr[:n] == prefix:
        return mystr[n:]

def bundle_biml_episode(S):
    # Bundle components for an episode suitable for optimizing BIML-S
    #  Pair each query 'xq' with all possible study examples
    ns = len(S['xs'])
    nq = len(S['xq'])
    S['xq+xs+ys'] = []
    for q_idx in range(nq):
        myquery = S['xq'][q_idx]
        S['xq+xs+ys'].append([myquery + [ITEM_token] + S['xs'][s_idx] + [IO_SEP] + S['ys'][s_idx] for s_idx in range(ns)])
    # S['xq+xs+ys'] is nq-length list of lists, each of which is a query paired with all possible study examples
    return S

def make_biml_batch(samples, langs):
    # Batch episodes into a series of padded source and target tensors.
    #   Each query is divided into several padded sources, one for each of the ns study examples
    # 
    # Input
    #  samples : list of dicts from bundle_biml_episode
    #  langs : input and output version of Lang class
    assert isinstance(samples,list)
    m = len(samples)
    mybatch = {}
    mybatch['list_samples'] = samples
    mybatch['batch_size'] = m    
    mybatch['xq'],mybatch['yq'],mybatch['xs'],mybatch['ys'] = [], [], [], []
    mybatch['q_idx'] = [] # index of which episode each query belongs to
    mybatch['s_idx'] = []# index of which episode each support belongs to
    mybatch['in_support'] = [] # bool list indicating whether each query is in its corresponding support set, or not
    mybatch['xq+xs+ys'] = []
    for idx in range(m):
        sample = samples[idx]
        nq = len(sample['xq'])
        ns = len(sample['xs'])
        assert(nq == len(sample['yq']))
        mybatch['xq'] += sample['xq']
        mybatch['yq'] += sample['yq']
        mybatch['xq+xs+ys'] += sample['xq+xs+ys']
        mybatch['q_idx'] += [idx*torch.ones(nq, dtype=torch.int)]
        mybatch['in_support'] += [x in sample['xs'] for x in sample['xq']]
        mybatch['xs'] += sample['xs']
        mybatch['ys'] += sample['ys']        
        mybatch['s_idx'] += [idx*torch.ones(ns, dtype=torch.int)]
    mybatch['q_idx'] = torch.cat(mybatch['q_idx'], dim=0)
    mybatch['s_idx'] = torch.cat(mybatch['s_idx'], dim=0)
    mybatch['yq_padded'],mybatch['yq_lengths'] = build_padded_tensor(mybatch['yq'], langs['output'])
    mybatch['yq_sos_padded'],mybatch['yq_sos_lengths'] = build_padded_tensor(mybatch['yq'],langs['output'],add_eos=False,add_sos=True)
    
    list_q_minibatch_padded, list_q_minibatch_lengths, list_q_minibatch_masks = [], [], []
    max_len_q_pairs = max([max([len(seq) for seq in query_w_support_pairs]) for query_w_support_pairs in mybatch['xq+xs+ys']])
    mybatch['xq+eos_padded'],mybatch['xq+eos_lengths'] = build_padded_tensor(mybatch['xq'], langs['input'], # m*nq x max_len_q_pairs
                                                                    max_len=max_len_q_pairs, add_eos=True)

    for query_w_support_pairs in mybatch['xq+xs+ys']: # for each query, get the list of that query paired with all ns study items
        q_minibatch_padded, q_minibatch_lengths = build_padded_tensor(query_w_support_pairs, langs['input'], # ns x max_len_q_pairs
                                                                    max_len=max_len_q_pairs, add_eos=False, add_sos=False) 
        q_mask = [(item[-2]==ITEM_token and item[-1]==IO_SEP) for item in query_w_support_pairs]
            # True for cases we want to ignore, where we don't have a support/query pairing (empty support)
        if all(q_mask): q_mask[0] = False # We must have one active pattern per query
        list_q_minibatch_padded.append(q_minibatch_padded)
        list_q_minibatch_lengths.append(torch.tensor(q_minibatch_lengths))
        list_q_minibatch_masks.append(torch.tensor(q_mask, dtype=bool))
    mybatch['xq+xs+ys_padded'] = torch.stack(list_q_minibatch_padded) # m*nq x ns x max_len_q_pairs
    mybatch['xq+xs+ys_lengths'] = torch.stack(list_q_minibatch_lengths) # m*nq x ns
    mybatch['xq+xs+ys_ignore_mask'] = torch.stack(list_q_minibatch_masks) # m*nq x ns
    return mybatch

def set_batch_to_device(batch):
    # Make sure all padded tensors are on GPU if needed
    tensors_to_gpu = [k for k in batch.keys() if ('_padded' in k or 'ignore_mask' in k)]
    for k in tensors_to_gpu:
        batch[k] = batch[k].to(device=DEVICE)
    return batch

def build_padded_tensor(list_seq, lang, add_eos=True, add_sos=False, max_len=-1):
    # Transform list of python lists to a padded torch tensors
    # 
    # Input
    #  list_seq : list of n sequences (each sequence is a python list of symbols)
    #  lang : language object for translation into indices
    #  pad_length : the maximum size of any sequence, including special characters
    #  add_eos : add end-of-sequence token at the end?
    #  add_sos : add start-of-sequence token at the beginning?
    #  max_len : default=-1, set length of padded sequences manually
    #
    # Output
    #  z_padded : LongTensor (n x max_len)
    #  z_lengths : python list of sequence lengths (list of scalars)
    n = len(list_seq)
    if n==0: return [],[]
    z_eos = deepcopy(list_seq)
    if add_sos: 
        z_eos = [[SOS_token]+z for z in z_eos]
    if add_eos:
        z_eos = [z+[EOS_token] for z in z_eos]    
    z_lengths = [len(z) for z in z_eos]
    if max_len<0:
        max_len = max(z_lengths) # maximum length in this episode
    else:
        assert(max_len) >= max(z_lengths), "max_len variable should be longer than any data sequence"
    z_padded = [pad_seq(z, max_len) for z in z_eos]
    z_padded = [lang.symbols_to_tensor(z, add_eos=False).unsqueeze(0) for z in z_padded]
    z_padded = torch.cat(z_padded, dim=0) # n x max_len
    return z_padded,z_lengths

def pad_seq(seq, max_length):
    # Pad sequence with the PAD_token symbol to achieve max_length
    #
    # Input
    #  seq : list of symbols
    #
    # Output
    #  seq : padded list now extended to length max_length
    seq += (max_length - len(seq)) * [PAD_token]
    return seq

def combine_input_output_symb(list_input_symb,list_output_symb):
    # Make new source vocabulary that combines list of input and output symbols.
    #  Include input/output and item separators, IO_SEP,ITEM_SEP.
    #  Exclude EOS_token,SOS_token,PAD_token, which will be added automatically by Lang constructor/
    #
    # Input
    #   list_input_symb : list of token symbols (strings)
    #   list_output_symb : list of token symbols (strings)
    # Output
    #   comb : combined list of tokens as strings
    assert(len(set(list_input_symb).intersection(set(list_output_symb)))==0), "in BIML, input and output vocabs should *not* share tokens"
    additional_symb = sorted(set([IO_SEP,ITEM_token])-set([EOS_token,SOS_token,PAD_token]))
    comb = sorted(set(list_input_symb + list_output_symb + additional_symb))
    return comb

def flip(p_head=0.5):
    # return True with probability p_head
    return random.random() < p_head

class DataSCAN(Dataset):
    # Meta-training and evaluation on SCAN for BIML
    
    def __init__(self, scan_type, mode, remap_type, ns=my_ns_default, inc_support_in_query=True, p_shuffle=0.95):
        # Input
        #   scan_type : name of scan split
        #   mode : 'train' or 'test'
        #   remap_type : which tokens to permute?  ['full','prims','actions','NONE']
        #   ns : number of support items 
        #   inc_support_in_query : (default=False) include all support examples as queries? If True, it messes up the log-likelihood calculations.
        #   p_shuffle : (default=0.95), probability that any meta-training episode uses permuted meanings
        assert mode in ['train','test']
        assert remap_type in ['full','prims','actions','NONE']

        # select SCAN task
        if scan_type == 'addprim_jump':
            fn = {'train':'SCAN/add_prim_split/tasks_train_addprim_jump.txt',
                  'test_support':'SCAN/add_prim_split/tasks_train_addprim_jump.txt', 
                  'test_query':'SCAN/add_prim_split/tasks_test_addprim_jump.txt'}
            self.test_support_manual_include = (['jump'],['I_JUMP'])
        elif scan_type == 'addprim_left':
            fn = {'train':'SCAN/add_prim_split/tasks_train_addprim_turn_left.txt',
                  'test_support':'SCAN/add_prim_split/tasks_train_addprim_turn_left.txt', 
                  'test_query':'SCAN/add_prim_split/tasks_test_addprim_turn_left.txt'}
            self.test_support_manual_include = (['turn','left'],['I_TURN_LEFT'])
        elif scan_type == 'around_right':
            fn = {'train':'SCAN/template_split/tasks_train_template_around_right.txt',
                  'test_support':'SCAN/template_split/tasks_train_template_around_right.txt', 
                  'test_query':'SCAN/template_split/tasks_test_template_around_right.txt'}
            self.test_support_manual_include = []
        elif scan_type == 'simple':
            fn = {'train':'SCAN/simple_split/tasks_train_simple.txt',
                  'test_support':'SCAN/simple_split/tasks_train_simple.txt',
                  'test_query':'SCAN/simple_split/tasks_test_simple.txt'}
            self.test_support_manual_include = []
        elif scan_type == 'opposite_right':
            fn = {'train':'SCAN/template_split/tasks_train_template_opposite_right.txt',
                  'test_support':'SCAN/template_split/tasks_train_template_opposite_right.txt', 
                  'test_query':'SCAN/template_split/tasks_test_template_opposite_right.txt'}
            self.test_support_manual_include = []
        elif scan_type == 'length':
            fn = {'train':'SCAN/length_split/tasks_train_length.txt',
                  'test_support':'SCAN/length_split/tasks_train_length.txt',
                  'test_query':'SCAN/length_split/tasks_test_length.txt'}
            self.test_support_manual_include = []
        else:
            assert False, "invalid SCAN type"

        # Select type of augmentation
        if remap_type == 'full':
            self.WI = input_symbols_list_default
            self.WO = output_symbols_list_default
        elif remap_type == 'prims': 
            self.WI = ['look','run','jump','walk']
            self.WO = ['I_LOOK','I_RUN','I_JUMP','I_WALK']
        elif remap_type == 'actions':
            self.WI = ['look', 'run', 'jump', 'walk', 'left', 'right']
            self.WO = ['I_LOOK','I_RUN','I_JUMP','I_WALK', 'I_TURN_LEFT', 'I_TURN_RIGHT']
        elif remap_type == 'NONE':
            self.WI = []
            self.WO = []
        else:
            assert False, "invalid remap type"
        
        self.ns = ns
        self.mode = mode
        self.is_train = mode == 'train'
        self.inc_support_in_query = inc_support_in_query
        self.p_shuffle = p_shuffle
        self.remap_type = remap_type
        if self.is_train:
            print('   Loading',fn['train'],'for training...')
            self.commands_support, self.outputs_support = readfile_scan(fn['train'])
            self.commands_query = deepcopy(self.commands_support)
            self.outputs_query = deepcopy(self.outputs_support)
        else:
            print('   Loading',fn['test_support'],'for testing support...')
            print('   Loading',fn['test_query'],'for testing query...')
            self.commands_support, self.outputs_support = readfile_scan(fn['test_support'])
            self.commands_query, self.outputs_query = readfile_scan(fn['test_query'])
            assert(remap_type=='NONE')
        print('    with num_support=',self.ns)
        self.pairs_support = list(zip(self.commands_support,self.outputs_support))
        self.pairs_query = list(zip(self.commands_query,self.outputs_query))
        self.maxlen_source = max([len(c) for c in self.commands_query + self.commands_support])
        self.maxlen_target = max([len(o) for o in self.outputs_query + self.outputs_support])+1
        print('    max input len',self.maxlen_source)
        print('    max output len',self.maxlen_target)
        self.input_symbols = input_symbols_list_default
        self.output_symbols = output_symbols_list_default
        comb = combine_input_output_symb(self.input_symbols,self.output_symbols)
        self.langs = {'input' : Lang(comb), 'output': Lang(self.output_symbols)}

    def __len__(self):
        return len(self.pairs_query)

    def __getitem__(self, idx):
        S = {}
        query1 = self.pairs_query[idx]
        S['xq'], S['yq'] = [query1[0]], [query1[1]]
        if self.is_train:
            assert(self.pairs_query[idx] == self.pairs_support[idx]), "during training, support and query distribution should be the same"
            pairs_support = random.sample(self.pairs_support[:idx]+self.pairs_support[idx+1:], self.ns) # exclude xq
        else:            
            pairs_support = random.sample(self.pairs_support, self.ns)
            if self.test_support_manual_include and (self.test_support_manual_include not in pairs_support):
                pairs_support[random.randint(0,self.ns-1)] = self.test_support_manual_include
        S['xs'], S['ys'] = [p[0] for p in pairs_support], [p[1] for p in pairs_support]
        S['xs'], S['ys'] = utils.shuffle(S['xs'], S['ys'])
        S['aux'] = {}
        S['aux']['ns'] = self.ns
        if self.remap_type!='NONE':
            if flip(self.p_shuffle):
                S = make_rand_permutation(S, self.WI, self.WO)
        if self.inc_support_in_query:
            s_choice = random.randint(0,self.ns-1)
            S['xq'] = [S['xs'][s_choice]] + S['xq']
            S['yq'] = [S['ys'][s_choice]] + S['yq']
        S = bundle_biml_episode(S)
        return S

class VanillaDataSCAN(Dataset):
    # Training and evaluation on SCAN for vanilla/basic seq2seq
    
    def __init__(self, scan_type, mode):
        # Input
        #   scan_type : name of scan split
        #   mode : 'train' or 'test'
        #   inc_support_in_query : (default=False) include all support examples as queries? If True, it messes up the log-likelihood calculations.
        assert mode in ['train','test']
        if scan_type == 'addprim_jump':
            fn = {'train':'SCAN/add_prim_split/tasks_train_addprim_jump.txt',
                  'test':'SCAN/add_prim_split/tasks_test_addprim_jump.txt'}
        elif scan_type == 'addprim_left':
            fn = {'train':'SCAN/add_prim_split/tasks_train_addprim_turn_left.txt',                  
                  'test':'SCAN/add_prim_split/tasks_test_addprim_turn_left.txt'}
        elif scan_type == 'around_right':
            fn = {'train':'SCAN/template_split/tasks_train_template_around_right.txt',
                  'test':'SCAN/template_split/tasks_test_template_around_right.txt'}
        elif scan_type == 'simple':
            fn = {'train':'SCAN/simple_split/tasks_train_simple.txt',
                  'test':'SCAN/simple_split/tasks_test_simple.txt'}
        elif scan_type == 'opposite_right':
            fn = {'train':'SCAN/template_split/tasks_train_template_opposite_right.txt',
                  'test':'SCAN/template_split/tasks_test_template_opposite_right.txt'}
        elif scan_type == 'length':
            fn = {'train':'SCAN/length_split/tasks_train_length.txt',
                  'test':'SCAN/length_split/tasks_test_length.txt'}
        else:
            assert False, "invalid SCAN type"
        self.ns = 1
        self.mode = mode
        self.is_train = mode == 'train'
        if self.is_train:
            print('   Loading',fn['train'],'for training...')
            self.commands, self.outputs = readfile_scan(fn['train'])
        else:
            print('   Loading',fn['test'],'for testing...')
            self.commands, self.outputs = readfile_scan(fn['test'])
        self.pairs = list(zip(self.commands,self.outputs))
        self.input_symbols = input_symbols_list_default
        self.output_symbols = output_symbols_list_default
        comb = combine_input_output_symb(self.input_symbols,[])
        self.langs = {'input' : Lang(comb), 'output': Lang(self.output_symbols)}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        S = {}
        query1 = self.pairs[idx]
        S['xq'], S['yq'] = [query1[0]], [query1[1]]
        S['xs'], S['ys'] = [[]], [[]]
        S['aux'] = {}
        S['aux']['ns'] = self.ns
        S = bundle_biml_episode(S)
        return S

def get_COGS_vocab(make_output_upper=True, filenames=['COGS/train.tsv','COGS/gen.tsv','COGS/test.tsv','COGS/dev.tsv']):
    # Get source and target vocabulary for COGS
    #
    # Input
    #  make_output_upper : convert output tokens to upper case
    #  filenames : list of filenames
    # 
    # Output
    #   source_vocab : list of tokens
    #   target_vocab : list of tokens
    source_vocab = set()
    target_vocab = set()
    for f in filenames:
        commands, outputs, _ = readfile_cogs(f, make_output_upper=make_output_upper)
        source_vocab.update(*commands)
        target_vocab.update(*outputs)
    source_vocab = sorted(list(source_vocab))
    target_vocab = sorted(list(target_vocab))
    return source_vocab, target_vocab

def get_COGS_common(to_upper=False):
    # Return list of all common nouns in COGS
    #     if to_upper=True, convert list to uppercase
    with open('COGS/vocab_common_nouns.txt','r') as fid:
        words = fid.readlines()
    words = [w.rstrip('\n').strip() for w in words]
    words = [w for w in words if len(w)>0]
    if to_upper: words = make_list_upper(words)
    return words

def get_COGS_proper(to_upper=False):
    # Return list of all proper nouns in COGS
    #     if to_upper=True, convert list to uppercase
    with open('COGS/vocab_proper_nouns.txt','r') as fid:
        words = fid.readlines()
    words = [w.rstrip('\n').strip() for w in words]
    words = [w for w in words if len(w)>0]
    if to_upper: words = make_list_upper(words)
    return words

def get_COGS_inf(to_upper=False):
    # Return list of all infinitive verbs in COGS
    #     if to_upper=True, convert list to uppercase
    with open('COGS/vocab_verbs_infinitive.txt','r') as fid:
        words = fid.readlines()
    words = [w.rstrip('\n').strip() for w in words]
    words = [w for w in words if len(w)>0]
    if to_upper: words = make_list_upper(words)
    return words

def get_COGS_dative(mytype, to_upper=False):
    # Return list of all dative verbs in COGS
    #     if to_upper=True, convert list to uppercase
    # Input
    #  mytype : either 'input' or 'output'
    if mytype=='input':
        with open('COGS/vocab_verbs_dative_input.txt','r') as fid:
            words = fid.readlines()
        assert not to_upper
    elif mytype=='output':
        with open('COGS/vocab_verbs_dative_output.txt','r') as fid:
            words = fid.readlines()
        assert to_upper
    words = [w.rstrip('\n').strip() for w in words]
    words = [w for w in words if len(w)>0]
    if to_upper: words = make_list_upper(words)
    return words

def get_max_flags(pairs, words_flag):
    # Compute the maximum number of flagged words (in list "words_flag") for all commands in the corpus 
    #
    # Input
    #   pairs : list of (command,output) pairs, where each in the pair is a list of words
    #   words_flag : list of special words to flag
    max_flag = -1
    all_count_flags = []
    for p in pairs:
        command = p[0]
        myflags = len([w for w in command if w in words_flag])
        all_count_flags.append(myflags)
        if myflags > max_flag:
            max_flag = myflags
    return max_flag

def common_to_proper_input(command_list, token_curr, token_new):
    # Given an input sentence, replace a common noun token with a proper noun token.
    #  This requires some structural sensitivity; e.g., removing the "a" or "the"
    # 
    # Input
    #  command_list : command in list form
    #  token_curr : (string) common noun
    #  token_new : (string) proper noun
    command_list = deepcopy(command_list)
    assert(isinstance(command_list,list))
    assert(isinstance(command_list[0],str))
    assert(token_curr in command_list), "current token must exist"
    idx = command_list.index(token_curr)
    command_list[idx] = token_new
    if len(command_list)>1: # if this is a sentence rather than a primitive definition
        assert(command_list[idx-1].lower() in ['a','the'])
        del command_list[idx-1]
    return command_list

def common_to_proper_output(output_list, token_curr, token_new):
    # Given an output expression, replace a common noun with a proper noun.
    #  This requires some structural sensitivity; e.g., removing the common noun *noun(x) and noun(x) nodes.
    #  Also, all variable indices after the commoun noun variable should be decremented 
    #
    # Input
    #  output_list : output in list form
    #  token_curr : (string) common noun
    #  token_new : (string) proper noun
    output_list = deepcopy(output_list)
    assert(isinstance(output_list,list))
    assert(isinstance(output_list[0],str))
    assert(token_curr in output_list), "token_curr must be in output_list"
    output_str = ' '.join(output_list)
    assert(token_curr.isupper())
    assert(token_new.isupper())
    matches = [ # re.match applies at beginning of string only; re.search applies anywhere
                re.match("LAMBDA &A &. " + token_curr + " \( &A \)", output_str), # flag for common noun primitive
                re.search("\* " + token_curr + " \( X _ (\d+) \) ;", output_str), # flag for common noun with 'the'
                re.match(token_curr + " \( X _ (\d+) \) AND ", output_str), # flag for common noun as *subject* with "a" and
                                                                            # NO other "the" determiner in sentence
                re.search("; " + token_curr + " \( X _ (\d+) \) AND ", output_str),# flag for common noun as *subject* with "a" and
                                                                                  # with >=1 "the" determiner in sentence
                re.search(" AND " + token_curr + " \( X _ (\d+) \)", output_str) # flag for common noun with "a" determiner
              ]
    filler_for_each_match = [' ', ' ', ' ', '; ', ' ']
    assert(sum([bool(m) for m in matches])==1), "there should be exactly one template that is matching"
    j = [bool(m) for m in matches].index(True)
    m = matches[j] # get regex matching structure
    if j==0: return [token_new] # if primitive common noun, return as primitive proper noun
    idx_start,idx_end = m.span()
    output_str = output_str[:idx_start] + filler_for_each_match[j] + output_str[idx_end:] # replace match's span with filler
    output_str = ' '.join(output_str.strip().split()) # tokenize and construct string again to remove extra spaces
    assert(len(m.groups())==1)
    my_var_idx = m.group(1)
    assert(my_var_idx.isnumeric())
    my_var_idx = int(my_var_idx)
    output_str = output_str.replace(" X _ "+str(my_var_idx)+" "," "+token_new+" ")
    output_str = output_str.replace(" "+token_curr+" "," "+token_new+" ")
    output_list = output_str.split()
    for j in range(len(output_list)):
        if output_list[j].isnumeric() and  int(output_list[j]) > my_var_idx:
            output_list[j] = str(int(output_list[j])-1)
    return output_list

class DataCOGS(Dataset):
    # Meta-training and evaluation on COGS for BIML

    def __init__(self, mode, remap_type, support_type='targeted', ns=my_ns_default, inc_support_in_query=True, p_shuffle=0.95, p_make_all_proper=0.01):
        # Input
        #   mode : specifies split to use for queries ['train','test_iid','gen_lexical','gen_structural','dev','dev_tiny']
        #   remap_type : specifies which words to permute ['targeted' (used in paper),'noun','NONE']
        #   support_type : (default=targeted) specifies which words to use to find support examples (usually the same as remap_type)
        #   ns : number of support items per episode
        #   inc_support_in_query : (default=True) include all support examples as queries? If True, it messes up the log-likelihood calculations.
        #   p_shuffle : (default=0.95), probability that an episode uses permuted meanings
        #   p_make_all_proper : (default=0.01), (only relevant to remap_type='targeted') probability that a common noun gets remapped to a proper noun
        assert mode in ['train','test_iid','gen_lexical','gen_structural','dev','dev_tiny']
        assert remap_type in ['targeted','noun','NONE']
        assert support_type in ['targeted','noun']
        self.mode = mode
        self.remap_type = remap_type
        self.support_type = support_type
        if remap_type == 'noun':
            assert(support_type == 'noun')
            self.WI = {}
            self.WO = {} 
            self.WI['proper'] = get_COGS_proper()
            self.WO['proper'] = get_COGS_proper(to_upper=True)
            self.WI['common'] = get_COGS_common()
            self.WO['common'] = get_COGS_common(to_upper=True)
        elif remap_type in ['targeted']:
            assert(support_type == 'targeted')
            self.WI = {}
            self.WO = {} 
            self.WI['proper'] = get_COGS_proper()
            self.WO['proper'] = get_COGS_proper(to_upper=True)
            self.WI['common'] = get_COGS_common()
            self.WO['common'] = get_COGS_common(to_upper=True)
            self.WI['inf'] = get_COGS_inf() # infinitive
            self.WO['inf'] = get_COGS_inf(to_upper=True)
            self.WI['dative'] = get_COGS_dative('input')
            self.WO['dative'] = get_COGS_dative('output',to_upper=True)
        elif remap_type == 'NONE':
            self.WI = []
            self.WO = []
        else:
            assert False, "invalid remap type"
        if support_type == 'noun':
            self.support_flags = get_COGS_common()+get_COGS_proper()
        elif support_type == 'targeted':
            self.support_flags = get_COGS_common()+get_COGS_proper()+get_COGS_inf()+get_COGS_dative('input')
        else:
            assert False, "invalid support_type type" 
        self.ns = ns
        self.inc_support_in_query = inc_support_in_query                
        self.p_shuffle = p_shuffle
        self.p_make_all_proper = p_make_all_proper
        files_all = {'train':'COGS/train.tsv',
                    'test_iid':'COGS/test.tsv',
                    'gen_lexical':'COGS/gen_lexical.tsv',
                    'gen_structural':'COGS/gen_structural.tsv',
                    'dev':'COGS/dev.tsv',
                    'dev_tiny':'COGS/dev_tiny.tsv'}
        self.fn_query = files_all[mode]
        print('  Started loading DataCOGS dataset...')
        print('   Loading',files_all['train'],'for making support sets')
        self.commands_support, self.outputs_support, self.type_labels_support = readfile_cogs(files_all['train'], make_output_upper=True)
        print('   Loading',self.fn_query,'for making queries (MODE =',self.mode,')')
        print('   Permutation type is:',self.remap_type)
        print('   Support     type is:',self.support_type)
        if self.remap_type!='NONE':
            print('   For each episode..')
            print('     Prob. of shuffling meanings            :',self.p_shuffle)
            print('     Prob. of converting common to proper nouns:',self.p_make_all_proper)
        self.commands_query, self.outputs_query, self.type_labels_query = readfile_cogs(self.fn_query, make_output_upper=True)
        self.pairs_support = list(zip(self.commands_support,self.outputs_support))
        self.pairs_query = list(zip(self.commands_query,self.outputs_query))
        self.maxlen_source = max([len(c) for c in self.commands_query + self.commands_support])
        self.maxlen_target = max([len(o) for o in self.outputs_query + self.outputs_support])+1
        print('   max input len',self.maxlen_source)
        print('   max output len',self.maxlen_target)
        self.max_flags = get_max_flags(self.pairs_query,self.support_flags)
        print('   max flagged words in a query command:',self.max_flags)
        print('  Finished loading DataCOGS.')
        self.indx_support_by_flag = self.__create_support_index()
        assert(self.max_flags <= self.ns) # there must be room for at least one support example per flagged word in a query
        self.input_symbols, self.output_symbols = get_COGS_vocab(make_output_upper=True)
        if isinstance(self.WI,list):
            assert(all([w in self.input_symbols for w in self.WI])), "all tokens in permute set WI must also be in input vocab"
            assert(all([w in self.output_symbols for w in self.WO])), "all tokens in permute set WO must also be in output vocab"
        for command in self.commands_support+self.commands_query:
            assert(all([w in self.input_symbols for w in command])), "all input tokens must be in input vocab"
        for output in self.outputs_support+self.outputs_query:            
            assert(all([w in self.output_symbols for w in output])), "all output tokens must be in output vocab"
        comb = combine_input_output_symb(self.input_symbols,self.output_symbols)
        self.langs = {'input' : Lang(comb), 'output': Lang(self.output_symbols)}

    def __len__(self):
        return len(self.pairs_query)

    def __getitem__(self, idx):
        S = {}
        query1 = self.pairs_query[idx]
        S['xq'], S['yq'] = [query1[0]], [query1[1]]
        S['xs'], S['ys'], n_missing_support = self.create_support_set(query1)
        S['xs'], S['ys'] = utils.shuffle(S['xs'], S['ys'])
        S['aux'] = {}
        S['aux']['ns'] = self.ns
        S['aux']['n_missing_support'] = n_missing_support
        S['aux']['q_type'] = self.type_labels_query[idx]
        if n_missing_support > 0: # if a flagged word is missing a corresponding study example, remove the query
            S['xq'], S['yq'] = [], []
        if self.remap_type!='NONE':
            if flip(self.p_shuffle):

                # surface-level word remapping
                if isinstance(self.WI,list): # one perm group
                    S = make_rand_permutation(S, self.WI, self.WO)
                elif isinstance(self.WI,dict): # multiple perm groups
                    for key in self.WI:
                        S = make_rand_permutation(S, self.WI[key], self.WO[key])
                else:
                    assert False # self.WI and self.WO should be a list or dict

                # structural word remapping
                if self.remap_type=='targeted':
                    S = self.stochastic_common_to_proper(S)

        if self.inc_support_in_query:
            s_choice = random.randint(0,self.ns-1)
            S['xq'] = [S['xs'][s_choice]] + S['xq']
            S['yq'] = [S['ys'][s_choice]] + S['yq']
        S = bundle_biml_episode(S)
        return S

    def __create_support_index(self):
        # Return dict that maps a flagged word to the list of commands (as command/output pairs) that contain the word
        indx_support_by_flag = {}
        for f in self.support_flags:
            indx_support_by_flag[f] = [pair for pair in self.pairs_support if f in pair[0]]
        return indx_support_by_flag

    def create_support_set(self, pair_query):
        #  Choose study set for this query.
        #   For each flagged token in the query command, find a study example that also has this flagged token.
        #   Fill the rest of the study set with arbitrary examples from the study corpus.
        #
        # Input
        #   pair_query : (command,output) pairing for the target query, where command and output are lists of strings
        # Output
        #   xs, ys: a set of support examples (input and output sequences)
        #   n_missing_support : number of flagged words that didn't have a study example we could add
        command = pair_query[0]
        assert(isinstance(command,list) and isinstance(command[0],str))
        myflags = [w for w in command if w in self.support_flags]
        my_pairs_support = []
        for f in myflags:
            pairs_options = self.indx_support_by_flag[f]
            pairs_options = [p for p in pairs_options if p != pair_query]
            if pairs_options:
                my_pairs_support.append(random.choice(pairs_options))
        n_missing_support = len(myflags) - len(my_pairs_support)
        my_pairs_support += random.sample(self.pairs_support,self.ns - len(my_pairs_support))
        xs = [p[0] for p in my_pairs_support]
        ys = [p[1] for p in my_pairs_support]
        assert(len(xs)==self.ns)
        return xs, ys, n_missing_support

    def stochastic_common_to_proper(self,S):
        # Convert a subset of common nouns in S to proper nouns. Make independent decisions for input and output sequences.
        #  For each possible remapping, see whether or not to convert the token via coin flip (p_convert = self.p_make_all_proper).
        #  Note that each remap requires a minimal structural transform in the output expression, or removing the determiner in the input sentence.
        unique_tokens_input =  sorted(list(set(sum(S['xs'] + S['xq'],[]))))
        unique_tokens_output = sorted(list(set(sum(S['ys'] + S['yq'],[]))))
        active_common_input  = [w for w in unique_tokens_input  if w in self.WI['common']]
        active_common_output = [w for w in unique_tokens_output if w in self.WO['common']]
        active_proper_input  = [w for w in unique_tokens_input  if w in self.WI['proper']]
        active_proper_output = [w for w in unique_tokens_output if w in self.WO['proper']]
        active_common_input  = [w for w in active_common_input if flip(self.p_make_all_proper)]
        active_common_output = [w for w in active_common_output if flip(self.p_make_all_proper)]
        if active_common_input:
            target_input  = random.sample(sorted(set(self.WI['proper'])-set(active_proper_input)), len(active_common_input)) # without replacement
            map_input = lambda command : self.remap_common_to_proper(command, active_common_input, target_input, common_to_proper_input)
            S['xs'] = list(map(map_input,S['xs']))
            S['xq'] = list(map(map_input,S['xq']))
        if active_common_output:
            target_output = random.sample(sorted(set(self.WO['proper'])-set(active_proper_output)), len(active_common_output))
            map_output = lambda output : self.remap_common_to_proper(output,  active_common_output, target_output, common_to_proper_output)
            S['ys'] = list(map(map_output,S['ys']))
            S['yq'] = list(map(map_output,S['yq']))
        return S
        
    def remap_common_to_proper(self,list_old,list_source,list_target,f_common_to_proper):
        # Convert all common nouns (in 'list_source') to the corresponding randomly chosen proper noun (in 'list_target').
        #
        # Input
        #  list_old : list of words where we will check each for a possible remap
        #  list_source : length k list of words to be replaced
        #  list_target : length k list of words that will replace the source words
        #  f_common_to_proper : function handle f(arg1,arg2,arg3) that takes list arg1 and maps all occurrences of string arg2 to the string arg3
        N = len(list_source)
        assert(len(list_target)==N)
        list_new = deepcopy(list_old)
        for j in range(N):
            if list_source[j] in list_old:
                list_new = f_common_to_proper(list_new, list_source[j], list_target[j])
        return list_new

class VanillaCOGS(Dataset):
    # Training and evaluation on COGS for basic/vanilla seq2seq 
    
    def __init__(self, mode):
        # Input
        #   scan_type : name of scan split
        #   mode : 'train' or 'test'
        #   inc_support_in_query : (default=False) include all support examples as queries? If True, it messes up the log-likelihood calculations.
        assert mode in ['train','test_iid','gen_lexical','gen_structural','dev','dev_tiny']
        self.ns = 1
        self.mode = mode
        self.is_train = mode == 'train'
        files_all = {'train':'COGS/train.tsv',
                    'test_iid':'COGS/test.tsv',
                    'gen_lexical':'COGS/gen_lexical.tsv',
                    'gen_structural':'COGS/gen_structural.tsv',
                    'dev':'COGS/dev.tsv',
                    'dev_tiny':'COGS/dev_tiny.tsv'}
        self.fn = files_all[mode]
        print('   Loading',self.fn,'for MODE =',self.mode)
        self.commands, self.outputs, self.type_labels = readfile_cogs(self.fn)
        self.maxlen_source = max([len(c) for c in self.commands])
        self.maxlen_target = max([len(o) for o in self.outputs])+1
        print('    max input len',self.maxlen_source)
        print('    max output len',self.maxlen_target)
        self.pairs = list(zip(self.commands,self.outputs))
        self.input_symbols, self.output_symbols = get_COGS_vocab()         
        comb = combine_input_output_symb(self.input_symbols,[])
        self.langs = {'input' : Lang(comb), 'output': Lang(self.output_symbols)}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        S = {}
        query1 = self.pairs[idx]
        S['xq'], S['yq'] = [query1[0]], [query1[1]]
        S['xs'], S['ys'] = [[]], [[]]
        S['aux'] = {}
        S['aux']['ns'] = self.ns
        S['aux']['q_type'] = self.type_labels[idx]
        S = bundle_biml_episode(S)
        return S

def make_rand_permutation(S, WI, WO):
    # Randomly permute certain tokens in the input and output sequences.
    #
    # Input
    #  S : dict with 'xs', 'ys', 'xq', 'yq' input/output examples in original form
    #  WI : list of input symbols we want to permute
    #  WO : list of output symbols we want to permute
    #
    # Output
    #   S : updated dict with remapped input/output patterns
    
    # construct mappings
    S_raw = deepcopy(S)
    assert(isinstance(WI,list) and isinstance(WO,list))
    while True: # ensure we don't have identity mapping
        WI_perm = deepcopy(WI)
        WO_perm = deepcopy(WO)
        random.shuffle(WI_perm)
        random.shuffle(WO_perm)
        if (WI_perm != WI or WO_perm != WO):
            break
    map_input = lambda command: list_remap(command,WI,WI_perm)
    map_output = lambda actions: list_remap(actions,WO,WO_perm)

    # functions to invert mappings
    unmap_input = lambda command: list_remap(command,WI_perm,WI)
    unmap_output = lambda actions: list_remap(actions,WO_perm,WO)

    S['aux'] = {}
    S['aux']['unmap_input'] = unmap_input
    S['aux']['unmap_output'] = unmap_output
    S['xs'] = list(map(map_input,S['xs']))
    S['xq'] = list(map(map_input,S['xq']))
    S['ys'] = list(map(map_output,S['ys']))
    S['yq'] = list(map(map_output,S['yq']))
    return S

if __name__ == "__main__":

    # Example episode for meta-training with full BIML model on COGS
    D_train, D_val = get_dataset('cogs_train_targeted')
    sample = D_train[0]
    print("")
    print('Example episode')
    print("")
    print('*Study examples*')
    display_input_output(sample['xs'],sample['ys'],sample['ys'])
    print("")
    print('*Query examples*')
    display_input_output(sample['xq'],sample['yq'],sample['yq'])