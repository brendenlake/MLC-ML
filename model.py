import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from copy import deepcopy
import matplotlib.pyplot as plt
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def describe_model(net):
    nparams = sum(p.numel() for p in net.parameters() if p.requires_grad)
    if type(net) is BIML_S:
        print('\nBIML_S specs:')
        print(' nparams=',nparams)
        print(' nlayers_encoder=',net.nlayers_encoder)
        print(' nlayers_decoder=',net.nlayers_decoder)
        print(' nhead=',net.nhead)
        print(' hidden_size=',net.hidden_size)
        print(' dim_feedforward=',net.dim_feedforward)
        print(' act_feedforward=',net.act)
        print(' dropout=',net.dropout_p)
        print(' ')
        print('')
    else:
        print('Network type ' + str(type(net)) + ' not found...')

class PositionalEncoding(nn.Module):
    #
    # Adds positional encoding to the token embeddings to introduce word order
    #
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000.) / emb_size) # size emb_size/2
        pos = torch.arange(0, maxlen).reshape(maxlen, 1) # maxlen x 1
        pos_embedding = torch.zeros((maxlen, emb_size)) # maxlen x emb_size
        pos_embedding[:, 0::2] = torch.sin(pos * den) # maxlen x emb_size/2
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2) # maxlen x 1 x emb_size
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        #  Input
        #    token_embedding: [seq_len, batch_size, embedding_dim] list of embedded tokens
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class BIML_S(nn.Module):
    #
    # BIML-Scale transformer trained for meta seq2seq learning
    #
    def __init__(self, hidden_size: int, input_size: int, output_size: int,
        PAD_idx_input: int, PAD_idx_output: int, ITEM_idx : int,
        nlayers_encoder: int=3, nlayers_decoder: int=3, nhead: int=8,
        dropout_p: float=0.1, ff_mult: int=4, activation='gelu', max_ns: int=50):
        #
        # Input        
        #  hidden_size : embedding size
        #  input_size  : number of input symbols
        #  output_size : number of output symbols
        #  PAD_idx_input : index of padding in input sequences
        #  PAD_idx_output : index of padding in output sequences
        #  ITEM_idx : index of token that is a placeholder for an embedded support example
        #  nlayers_encoder : number of transformer encoder layers
        #  nlayers_decoder : number of transformer decoder layers (likely fewer than encoder for tasks with deterministic outputs)
        #  nhead : number of heads for multi-head attention
        #  dropout_p : dropout applied to symbol embeddings and transformer layers
        #  ff_mult : multiplier for hidden size of feedforward network
        #  activation: string either 'gelu' or 'relu'
        #  max_ns : maximum number of study examples
        super(BIML_S, self).__init__()
        assert activation in ['gelu','relu']
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.PAD_idx_input = PAD_idx_input
        self.PAD_idx_output = PAD_idx_output
        self.ITEM_idx = ITEM_idx
        self.nlayers_encoder = nlayers_encoder
        self.nlayers_decoder = nlayers_decoder
        self.nhead = nhead
        self.dropout_p = dropout_p
        self.dim_feedforward = hidden_size*ff_mult
        self.act = activation
        self.max_ns = max_ns
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=self.dim_feedforward,
            dropout=dropout_p, batch_first=True, activation=activation)
        self.pair_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers_encoder)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=self.dim_feedforward,
            dropout=dropout_p, batch_first=True, activation=activation)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayers_decoder)
        self.positional_encoding = PositionalEncoding(emb_size=hidden_size, dropout=dropout_p)
        self.input_embedding = nn.Embedding(input_size, hidden_size)
        self.output_embedding = nn.Embedding(output_size, hidden_size)
        self.study_example_embedding = nn.Embedding(self.max_ns, hidden_size)
        self.out = nn.Linear(hidden_size,output_size)

    def prep_decode(self, z_padded):
        # Embed target sequences and make masks 
        #
        # Input
        #  z_padded : b*nq (batch_size) x maxlen_tgt (padded target sequences)
        #  z_lengths : b*nq list (actual sequence lengths)
        maxlen_tgt = z_padded.size(1)
        z_embed = self.output_embedding(z_padded) # batch_size x maxlen_tgt x emb_size

        # Add positional encoding to target embeddings
        tgt_embed = self.positional_encoding(z_embed.transpose(0,1))
        tgt_embed = tgt_embed.transpose(0,1) # batch_size x maxlen_tgt x emb_size

        # create mask for padded targets
        tgt_padding_mask = z_padded==self.PAD_idx_output # batch_size x maxlen_tgt
            # value of True means ignore

        # create diagonal mask for autoregressive control
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(maxlen_tgt) # maxlen_tgt x maxlen_tgt
        tgt_mask = tgt_mask.to(device=DEVICE)
        return tgt_embed, tgt_padding_mask, tgt_mask

    def forward(self, z_padded, batch):
        # Forward pass through encoder and decoder
        # 
        # Input
        #  z_padded : tensor of size [b*nq (batch_size), maxlen_target] : decoder input via token index
        #  batch : struct via datasets.make_biml_batch(), which includes source sequences
        # 
        # Output
        #   output : [b*nq x maxlen_target x output_size]
        memory, memory_padding_mask = self.encode(batch)
        return self.decode(z_padded, memory, memory_padding_mask)

    def encode(self, batch):
        # Forward pass through encoders only
        #   Note, "batch_size" is b*nq, or original batch size * number of queries
        #
        # Output
        #  memory : [b*nq (batch_size) x maxlen_src x emb_size]
        #  memory_padding_mask : [b*nq (batch_size) x maxlen_src] binary mask
        #        
        # Step 1: Simple embeddings for each pairing of a query xq[i] with one of its support items, (xs -> ys)[j] for j=1...ns
        pairs_padded = batch['xq+xs+ys_padded'] # batch_size x ns x maxlen_pairs
        pairs_ignore_mask = batch['xq+xs+ys_ignore_mask'] # batch_size x ns (True for empty xq/xs pairs)
        is_complete_pairing = not torch.any(pairs_ignore_mask) # in this case, all episodes have same number of study examples
        xq_eos_padded = batch['xq+eos_padded'] # batch_size x maxlen_pairs
        xq_eos_lengths = batch['xq+eos_lengths'] # batch_size
        pairs_embed = self.input_embedding(pairs_padded) # batch_size x ns x maxlen_pairs x emb_size
        sz = pairs_embed.shape
        assert(self.max_ns>=sz[1]), "cannot exceed self.max_ns number of study examples"
        
        # Step 2: Contextual embedding of each pairing between query and study examples
        pairs_padded_flat = torch.reshape(pairs_padded,(sz[0]*sz[1],sz[2])) # ns*batch_size x maxlen_pairs
        pairs_embed_flat = torch.reshape(pairs_embed,(sz[0]*sz[1],sz[2],sz[3])) # ns*batch_size x maxlen_pairs x emb_size        
        pairs_embed_flat = self.positional_encoding(pairs_embed_flat.transpose(0,1)).transpose(0,1)
            # ns*batch_size x maxlen_pairs x emb_size
        pairs_flat_padding_mask = pairs_padded_flat == self.PAD_idx_input # ns*batch_size x maxlen_pairs
        if is_complete_pairing:
            pairs_embed_flat = self.pair_encoder(pairs_embed_flat, src_key_padding_mask = pairs_flat_padding_mask)
                # ns*batch_size x maxlen_pairs x emb_size
        else:              
            pairs_ignore_mask_flat = torch.reshape(pairs_ignore_mask,(sz[0]*sz[1],)) # ns*batch_size
            pairs_keep_mask_flat = torch.logical_not(pairs_ignore_mask_flat) # (ns*batch_size) 1D tensor
            pairs_embed_flat[pairs_keep_mask_flat] = self.pair_encoder(pairs_embed_flat[pairs_keep_mask_flat], src_key_padding_mask = pairs_flat_padding_mask[pairs_keep_mask_flat])
                # ns*batch_size x maxlen_pairs x emb_size
        pairs_embed = torch.reshape(pairs_embed_flat,sz) # batch_size x ns x maxlen_pairs x emb_size          

        # Step 3: Add another embedding to indicate which study example is which
        example_idx = torch.arange(sz[1]).int() # ns
        example_idx = example_idx.to(device=DEVICE)
        example_idx = torch.reshape(example_idx,(1,sz[1],1)) # 1 x ns x 1
        example_idx = example_idx.expand(sz[:3]) # batch_size x ns x maxlen_pairs
        example_idx_embed = self.study_example_embedding(example_idx) # batch_size x ns x maxlen_pairs x emb_size 
        pairs_embed = pairs_embed + example_idx_embed # batch_size x ns x maxlen_pairs x emb_size

        # Step 4: Reshape embeddings as individual source sequences and create ignore_mask
        src_embed = torch.reshape(pairs_embed,(sz[0],sz[1]*sz[2],sz[3])) # batch_size x (ns*maxlen_pairs) x emb_size
        pairs_padding_ignore_mask = pairs_padded == self.PAD_idx_input # batch_size x ns x maxlen_pairs
        if not is_complete_pairing: pairs_padding_ignore_mask[pairs_ignore_mask] = True
        src_padding_ignore_mask = torch.reshape(pairs_padding_ignore_mask,(sz[0],sz[1]*sz[2])) # batch_size x (ns*maxlen_pairs)

        memory = src_embed  # batch_size x (ns*maxlen_pairs) x emb_size
        memory_padding_mask = src_padding_ignore_mask # batch_size x (ns x maxlen_pairs)
        return memory, memory_padding_mask

    def decode(self, z_padded, memory, memory_padding_mask):
        # Forward pass through decoder only
        #
        # Input
        # 
        #  memory : [b*nq (batch_size) x maxlen_src x hidden_size] output of transformer encoder
        #  memory_padding_mask : [b*nq (batch_size) x maxlen_src x hidden_size] binary mask padding where False means leave alone
        #
        # Output
        #   output : [b*nq x maxlen_target x output_size]
        tgt_embed, tgt_padding_mask, tgt_mask = self.prep_decode(z_padded)
            # batch_size x maxlen_tgt x emb_size
        trans_out = self.decoder(tgt_embed, memory,
                tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=memory_padding_mask)
        output = self.out(trans_out)
        return output