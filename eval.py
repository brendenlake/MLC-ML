import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical
import os
import sys
import argparse
import numpy as np
import math
from copy import deepcopy
from model import BIML_S, describe_model
import datasets as dat
from train_lib import seed_all, extract, display_input_output, assert_consist_langs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Evaluate a pre-trained model

def evaluate_ll(val_dataloader, net, langs, loss_fn=[], p_lapse=0.0, verbose=False):
    # Evaluate the total (sum) log-likelihood across the entire validation set
    # 
    # Input
    #   val_dataloader : 
    #   net : BIML-S model
    #   langs : dict of dat.Lang classes
    #   p_lapse : (default 0.) combine decoder outputs (prob 1-p_lapse) as mixture with uniform distribution (prob p_lapse)
    net.eval()
    total_N = 0
    total_ll = 0
    if not loss_fn: loss_fn = torch.nn.CrossEntropyLoss(ignore_index=langs['output'].PAD_idx)
    for batch_idx, val_batch in enumerate(val_dataloader):
        val_batch = dat.set_batch_to_device(val_batch)
        dict_loss = batch_ll(val_batch, net, loss_fn, langs, p_lapse=p_lapse)
        total_ll += dict_loss['ll']
        total_N += dict_loss['N']
    return total_ll, total_N

def evaluate_acc(val_dataloader, net, langs, max_length, eval_type='max', verbose=False):
    # Evaluate accuracy (exact match) across entire validation set
    #
    # Input
    #   val_dataloader : 
    #   net : BIML-S model
    #   langs : dict of dat.Lang classes
    #   max_length : maximum length of output sequences
    #   langs : dict of dat.Lang classes
    #   eval_type : 'max' for greedy decoding, 'sample' for sample from distribution
    net.eval()
    samples_pred = [] # list of all episodes with model predictions
    for batch_idx, val_batch in enumerate(val_dataloader): # each batch
        val_batch = dat.set_batch_to_device(val_batch)
        scores = batch_acc(val_batch, net, langs, max_length, eval_type=eval_type)        
        samples_batch = val_batch['list_samples']
        for sidx in range(len(samples_batch)): # for each episode of the batch
            yq_sel = val_batch['q_idx'].cpu().numpy() == sidx # select for queries in this episode
            in_support = scores['in_support'][yq_sel] #numpy array
            is_novel = np.logical_not(in_support)
            v_acc = scores['v_acc'][yq_sel] #numpy array
            samples_batch[sidx]['yq_predict'] = extract(yq_sel, scores['yq_predict'])
            samples_batch[sidx]['v_acc'] = v_acc            
            samples_batch[sidx]['in_support'] = in_support #numpy array
            samples_batch[sidx]['acc_retrieve'] = np.mean(v_acc[in_support])*100.
            samples_batch[sidx]['acc_novel'] = np.mean(v_acc[is_novel])*100.
        samples_pred.extend(samples_batch)

    # Compute mean accuracy across all val episodes
    mean_acc_retrieve = np.nanmean([sample['acc_retrieve'] for sample in samples_pred])
    v_acc_novel = [sample['acc_novel'] for sample in samples_pred]
    mean_acc_novel = np.mean(v_acc_novel)

    if verbose:
        display_console_pred(samples_pred)
    return {'samples_pred':samples_pred, 'mean_acc_novel':mean_acc_novel, 'mean_acc_retrieve':mean_acc_retrieve, 'v_novel':v_acc_novel}

def batch_ll(batch, net, loss_fn, langs, p_lapse=0.0):
    # Evaluate log-likelihood (average over cells, and sum total) for a given batch
    #
    # Input
    #   batch : from dat.make_biml_batch
    #   loss_fn : loss function
    #   langs : dict of dat.Lang classes
    net.eval()
    m = len(batch['yq']) # b*nq
    target_batches = batch['yq_padded'] # b*nq x max_length
    target_lengths = batch['yq_lengths'] # list of size b*nq
    target_shift = batch['yq_sos_padded'] # b*nq x max_length
        # Shifted targets with padding (added SOS symbol at beginning and removed EOS symbol) 
    decoder_output = net(target_shift, batch)
        # b*nq x max_length x output_size    

    logits_flat = decoder_output.reshape(-1, decoder_output.shape[-1]) # (batch*max_len, output_size)
    if p_lapse > 0:
        logits_flat = smooth_decoder_outputs(logits_flat,p_lapse,langs['output'].symbols+[dat.EOS_token],langs)
    loss = loss_fn(logits_flat, target_batches.reshape(-1))
    loglike = -loss.cpu().item()
    dict_loss = {}
    dict_loss['ll_by_cell'] = loglike # average over cells
    dict_loss['N'] = float(sum(target_lengths)) # total number of valid cells
    dict_loss['ll'] = dict_loss['ll_by_cell'] * dict_loss['N'] # total LL
    return dict_loss

def smooth_decoder_outputs(logits_flat,p_lapse,lapse_symb_include,langs):
    # Mix decoder outputs (logits_flat) with uniform distribution over allowed emissions (in lapse_symb_include)
    #
    # Input
    #  logits_flat : (batch*max_len, output_size) # unnomralized log-probabilities
    #  p_lapse : probability of a uniform lapse
    #  lapse_symb_include : list of tokens (strings) that we want to include in the lapse model
    #
    # Output
    #  log_probs_flat : (batch*max_len, output_size) normalized log-probabilities
    lapse_idx_include = [langs['output'].symbol2index[s] for s in lapse_symb_include]
    assert dat.SOS_token not in lapse_symb_include # SOS should not be an allowed output through lapse model
    sz = logits_flat.size() # get size (batch*max_len, output_size)
    probs_flat = F.softmax(logits_flat,dim=1) # (batch*max_len, output_size)
    num_classes_lapse = len(lapse_idx_include)
    probs_lapse = torch.zeros(sz, dtype=torch.float)
    probs_lapse = probs_lapse.to(device=DEVICE)
    probs_lapse[:,lapse_idx_include] = 1./float(num_classes_lapse)
    log_probs_flat = torch.log((1-p_lapse)*probs_flat + p_lapse*probs_lapse) # (batch*max_len, output_size)
    return log_probs_flat

def batch_acc(batch, net, langs, max_length, eval_type='max', out_mask_allow=[]):
    # Evaluate exact match accuracy for a given batch
    #
    #  Input
    #   batch : from dat.make_biml_batch
    #   net : BIML model
    #   max_length : maximum length of output sequences
    #   langs : dict of dat.Lang classes
    #   eval_type : 'max' for greedy decoding, 'sample' for sample from distribution
    #   out_mask_allow : default=[]; list of emission symbols (strings) we want to allow. Default of [] allows all output emissions
    assert eval_type in ['max','sample']
    net.eval()
    emission_lang = langs['output']
    use_mask = len(out_mask_allow)>0
    memory, memory_padding_mask = net.encode(batch) 
        # memory : b*nq x maxlength_src x hidden_size
        # memory_padding_mask : b*nq x maxlength_src (False means leave alone)
    m = len(batch['yq']) # b*nq
    z_padded = torch.tensor([emission_lang.symbol2index[dat.SOS_token]]*m) # b*nq length tensor
    z_padded = z_padded.unsqueeze(1) # [b*nq x 1] tensor
    z_padded = z_padded.to(device=DEVICE)
    max_length_target = batch['yq_padded'].shape[1]-1 # length without EOS
    assert max_length >= max_length_target # make sure that the net can generate targets of the proper length

    # make the output mask
    if use_mask:
        assert dat.EOS_token in out_mask_allow # EOS must be included as an allowed symbol
        additive_out_mask = -torch.inf * torch.ones((m,net.output_size), dtype=torch.float)
        additive_out_mask = additive_out_mask.to(device=DEVICE)
        for s in out_mask_allow:
            sidx = langs['output'].symbol2index[s]
            additive_out_mask[:,sidx] = 0.

    # Run through decoder
    all_decoder_outputs = torch.zeros((m, max_length), dtype=torch.long)
    all_decoder_outputs = all_decoder_outputs.to(device=DEVICE)
    for t in range(max_length):
        decoder_output = net.decode(z_padded, memory, memory_padding_mask)
            # decoder_output is b*nq x (t+1) x output_size
        decoder_output = decoder_output[:,-1] # get the last step's output (batch_size x output_size)
        if use_mask: decoder_output += additive_out_mask

        # Choose the symbols at next timestep
        if eval_type == 'max': # pick the most likely
            topi = torch.argmax(decoder_output,dim=1)
            emissions = topi.view(-1)
        elif eval_type == 'sample':
            emissions = Categorical(logits=decoder_output).sample()
        all_decoder_outputs[:,t] = emissions
        z_padded = torch.cat([z_padded, emissions.unsqueeze(1)], dim=1)
        if all_batch_has_eos(z_padded,emission_lang.symbol2index[dat.EOS_token]):
            break

    # Get predictions as strings and see if they are correct
    all_decoder_outputs = all_decoder_outputs.detach()
    yq_predict = [] # list of all predicted query outputs as strings
    v_acc = np.zeros(m)
    for q in range(m):
        myseq = emission_lang.tensor_to_symbols(all_decoder_outputs[q,:].view(-1))
        yq_predict.append(myseq)
        v_acc[q] = yq_predict[q] == batch['yq'][q]
    in_support = np.array(batch['in_support'])
    out = {'yq_predict':yq_predict, 'v_acc':v_acc, 'in_support':in_support}      
    return out

def all_batch_has_eos(z_padded,eos_index):
    # Returns True if each generated output sequence (row of z_apdded)
    #  has at least one eos token (indicated by eos_index)
    #
    # Input
    #  z_padded: [m x T] tensor of token integers
    #  eos_index: integer that represents an eos
    assert(z_padded.dim()==2)
    count_eos = torch.sum(z_padded==eos_index, dim=1)
    assert(z_padded.shape[0]==torch.numel(count_eos))
    return torch.all(count_eos>0)

def viz_train_dashboard(train_tracker):
    # Show loss curves
    import matplotlib.pyplot as plt
    if not train_tracker:
        print('No training stats to plot')
        return
    fv = lambda x : [t[x] for t in train_tracker]
    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.plot(fv('step'),fv('avg_train_loss'),'b',label='train')
    if 'val_loss' in train_tracker[0] : plt.plot(fv('step'),fv('val_loss'),'r',label='val')
    plt.ylim((0,.3))
    plt.xlabel('step')
    plt.legend()
    plt.title('Loss')
    plt.subplot(2, 2, 2)
    plt.plot(fv('step'),fv('lr'),'b')
    plt.xlabel('step')
    plt.title('Learning rate')
    plt.show()

def display_console_pred(samples_pred):
    # Show model predictions in console.
    #  If available, show predictions categorized by query types
    #
    # Input
    #  samples_pred : list of dicts from evaluate_acc, which has predictions for each episode
    has_qtype = 'aux' in samples_pred[0] and 'q_type' in samples_pred[0]['aux']
    acc_by_type = {}

    for idx,sample in enumerate(samples_pred):
        print('Evaluation episode ' + str(idx))
        in_support = sample['in_support']
        is_novel = np.logical_not(in_support)
        if 'grammar' in sample:
            print("")
            print(sample['grammar'])
        if has_qtype:
            qtype = sample['aux']['q_type']
            print('  trial type;',qtype)                
        print('  support items;')
        display_input_output(sample['xs'],sample['ys'],sample['ys'])
        print('  retrieval items;',round(sample['acc_retrieve'],3),'% correct')
        display_input_output(extract(in_support,sample['xq']),extract(in_support,sample['yq_predict']),extract(in_support,sample['yq']))
        print('  generalization items;',round(sample['acc_novel'],3),'% correct')
        display_input_output(extract(is_novel,sample['xq']),extract(is_novel,sample['yq_predict']),extract(is_novel,sample['yq']))
        if has_qtype:
            if qtype not in acc_by_type:
                acc_by_type[qtype] = [sample['acc_novel']]
            else:
                acc_by_type[qtype].append(sample['acc_novel'])

    if has_qtype: # print accuracy for each trial type
        acc_view = sorted( ((np.mean(v),k,len(v)) for k,v in acc_by_type.items()), reverse=True)
        print('Accuracy by query type:')
        for v,k,n in acc_view:
            print('  ',round(v,3),'(N=',n,') :',k)

if __name__ == "__main__":

        # Adjustable parameters
        parser = argparse.ArgumentParser()
        parser.add_argument('--fn_out_model', type=str, default='', help='*REQUIRED*. Filename for loading the model')
        parser.add_argument('--dir_model', type=str, default='out_models', help='Directory for loading the model file')
        parser.add_argument('--max_length_eval', type=int, default=50, help='Maximum generated sequence length (must be at least 50 for SCAN and 400 for COGS)')
        parser.add_argument('--batch_size', type=int, default=-1, help='Maximum generated sequence length')                                
        parser.add_argument('--episode_type', type=str, default='', help='What type of episodes do we want? See datasets.py for options')
        parser.add_argument('--dashboard', default=False, action='store_true', help='Showing loss curves during training.')
        parser.add_argument('--ll', default=False, action='store_true', help='Evaluate log-likelihood of validation (val) set')
        parser.add_argument('--max', default=False, action='store_true', help='Find best outputs for val commands (greedy decoding)')
        parser.add_argument('--debug', default=False, action='store_true')
        parser.add_argument('--verbose', default=False, action='store_true', help='Inspect outputs in more detail')

        args = parser.parse_args()
        fn_out_model = args.fn_out_model
        dir_model = args.dir_model
        episode_type = args.episode_type
        max_length_eval = args.max_length_eval        
        do_dashboard = args.dashboard
        batch_size = args.batch_size
        do_ll = args.ll
        do_max_acc = args.max
        do_debug = args.debug
        verbose = args.verbose

        model_tag = episode_type + '_' + fn_out_model.replace('.pt','')
        fn_out_model = os.path.join(dir_model, fn_out_model)
        if not os.path.isfile(fn_out_model):
             raise Exception('filename '+fn_out_model+' not found')

        seed_all()
        print('Loading model:',fn_out_model,'on',DEVICE)
        checkpoint = torch.load(fn_out_model, map_location=DEVICE)
        if not episode_type: episode_type = checkpoint['episode_type']
        if batch_size<=0: batch_size = checkpoint['batch_size']
        nets_state_dict = checkpoint['nets_state_dict']
        if list(nets_state_dict.keys())==['net']: nets_state_dict = nets_state_dict['net'] # for compatability with legacy code
        input_size = checkpoint['langs']['input'].n_symbols
        output_size = checkpoint['langs']['output'].n_symbols
        emb_size = checkpoint['emb_size']
        dropout_p = checkpoint['dropout']
        ff_mult = checkpoint['ff_mult']
        myact = checkpoint['activation']
        nlayers_encoder = checkpoint['nlayers_encoder']
        nlayers_decoder = checkpoint['nlayers_decoder']
        train_tracker = checkpoint['train_tracker']
        best_val_loss = -float('inf')
        if 'best_val_loss' in checkpoint: best_val_loss = checkpoint['best_val_loss']
            
        print(' Loading model that has completed (or started) ' + str(checkpoint['epoch']) + ' of ' + str(checkpoint['nepochs']) + ' epochs')
        print('  test episode_type:',episode_type)
        print('  batch size for train:',checkpoint['batch_size'])
        print('  batch size for test:',batch_size)        
        print('  number of steps:', checkpoint['step'])
        print('  best val loss achieved: {:.4f}'.format(best_val_loss))

        # Load validation dataset
        D_train,D_val = dat.get_dataset(episode_type)
        langs = D_val.langs
        assert_consist_langs(langs,checkpoint['langs'])
        train_dataloader = DataLoader(D_train,batch_size=batch_size,
                                    collate_fn=lambda x:dat.make_biml_batch(x,langs),shuffle=False)
        val_dataloader = DataLoader(D_val,batch_size=batch_size,
                                    collate_fn=lambda x:dat.make_biml_batch(x,langs),shuffle=False)

        # Set maximum length (if COGS, set it dynamically based on split)
        if 'cogs' in episode_type: max_length_eval = D_val.maxlen_target
        print('  max eval length:', max_length_eval)
        
        # Load model parameters         
        net = BIML_S(emb_size, input_size, output_size,
            langs['input'].PAD_idx, langs['output'].PAD_idx, langs['input'].ITEM_idx,
            nlayers_encoder=nlayers_encoder, nlayers_decoder=nlayers_decoder, 
            dropout_p=dropout_p, activation=myact, ff_mult=ff_mult)        
        net.load_state_dict(nets_state_dict)
        net = net.to(device=DEVICE)
        describe_model(net)
    
        # Perform selected evaluations
        if do_dashboard:
            print('Showing loss curves during training <close plot to continue>')
            viz_train_dashboard(train_tracker)
        if do_ll:
            seed_all()
            print('Evaluating log-likelihood of val episodes...')
            total_ll,total_N = evaluate_ll(val_dataloader, net, langs)
            print('evaluation on',episode_type,'loglike:',round(total_ll,4),'for',int(total_N),'symbol predictions')
            print('mean loglike is',round(total_ll/total_N,5),'per symbol')
        if do_max_acc:
            seed_all()
            E = evaluate_acc(val_dataloader, net, langs, max_length_eval, eval_type='max', verbose=verbose)
            print('Evaluating set of validation episodes (via greedy decoding)...')
            print(' Acc Retrieve (val):',round(E['mean_acc_retrieve'],4))
            print(' Acc Novel (val):',round(np.mean(E['v_novel']),4),
                    'SD=',round(np.std(E['v_novel']),4),'N=',len(E['v_novel']))