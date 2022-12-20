import random
import numpy as np
import torch
import time
import math
from copy import deepcopy, copy

def seed_all(seed=0):
    print("* Setting all random seeds to ",seed,'*')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def extract(include,arr):
    # Create a new list only using the included (boolean) elements of arr 
    #
    # Input
    #  include : [n len] boolean array (numpy or python)
    #  arr : [ n length array]
    assert len(include)==len(arr)
    return [a for idx,a in enumerate(arr) if include[idx]]

def display_input_output(input_patterns,output_patterns,target_patterns):
    # Verbose printing of query items
    # 
    # Input
    #   input_patterns : list of input sequences (query inputs; each sequence is in list form)
    #   output_patterns : list of predicted output sequences (query outputs; each sequence is in list form)
    #   target_patterns : list of targets (query outputs; each sequence is in list form)
    nq = len(input_patterns)
    if nq == 0:
        print('     no patterns')
        return
    for q in range(nq):
        assert isinstance(input_patterns[q],list)
        assert isinstance(output_patterns[q],list)
        is_correct = output_patterns[q] == target_patterns[q]        
        print('     ',end='')
        print(' '.join(input_patterns[q]),end='')
        print(' -> ',end='')
        print(' '.join(output_patterns[q]),end='')
        if not is_correct:
            print(' (** target: ',end='')
            print(' '.join(target_patterns[q]),end='')
            print(')',end='')
        print('')

# Robertson's asMinutes and timeSince helper functions to print time elapsed and estimated time
# remaining given the current time and progress

def asMinutes(s): 
    # convert seconds to minutes
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    # prints time elapsed and estimated time remaining
    #
    # Input 
    #  since : previous time
    #  percent : amount of training complete
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))  

def list_remap(list_old,list_source,list_target):
    #  For a given list, replace each token in "source" with the corresponding token in "target"
    # 
    # Input
    #  list_old : list of tokens where we will check each for a remap
    #  list_source : length k list of tokens to be replaced
    #  list_target : length k list of tokens that will replace the source tokens
    assert(len(list_source)==len(list_target))
    mydict = dict(zip(list_source,list_target))
    list_new = deepcopy(list_old)
    for i in range(len(list_new)):
        if list_new[i] in mydict:
            list_new[i] = mydict[list_new[i]]
    return list_new

def assert_consist_langs(langs_new,langs_old):
    # Make sure all symbols/indices in langs_old are the same as in langs_new
    for s in langs_old['input'].symbol2index.keys():
        assert(langs_old['input'].symbol2index[s] == langs_new['input'].symbol2index[s])
    for s in langs_old['output'].symbol2index.keys():
        assert(langs_old['output'].symbol2index[s] == langs_new['output'].symbol2index[s])