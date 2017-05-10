import torch.nn as nn
from torch.autograd import Variable

import torch
from torch.nn.modules.rnn import LSTMCell

from torch.nn import functional as F

# TODO
class UnsharedRNNModel(nn.Module):
    """
        WIP:
            so far, we only implement 1-directional LSTM (with dropout)

        Same as the RNNModel (below), except that we don't share weights across time

        In general, we could:
            share all the weights
            share only the non-recurrent weights <--- this is the one we do ATM
            share none of the weights

        The decision to tie the encoder / decoder weights is orthogonal to this choice
    """

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, ntsteps,
            dropout=0.5, tie_weights=False):#, weight_sharing='none'):
        super(UnsharedRNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(nhid, ntoken)
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        
        self.rnn = UnsharedRNN(ninp, nhid, nlayers, ntsteps, dropout=dropout)

        self.init_weights()

        assert rnn_type == 'LSTM'
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.ntsteps = ntsteps

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    # This is the whole forward prop; hidden is just the initial hiddens
    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

# modified from https://raw.githubusercontent.com/pytorch/pytorch/master/torch/nn/_functions/rnn.py 
def LSTMCell(gates, cc):
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)

    cc = (forgetgate * cc) + (ingate * cellgate)
    hh = outgate * F.tanh(cc)

    return hh,cc


# modified from https://discuss.pytorch.org/t/how-to-speed-up-for-loop-in-customized-rnn/1012
class UnsharedRNN(nn.Module):
    def __init__(self, ninp, nhid, nlayers, ntsteps, dropout=None):
        self.__dict__.update(locals())
        super(UnsharedRNN, self).__init__()
        # for some reason, these weights were torch.FloatTensor instead of torch.cuda.FloatTensor
        # I could try just putting everything inside the parent class...
        # I could try actually figuring out what's going on...
        # I could try manually moving things to GPU <--- I did this
        # FIXME: I think these are still not being recognized as parameters and trained!
        self.i2h = [ [nn.Linear(ninp + nhid, nhid * 4) for layer in range(self.nlayers)] for tstep in range(self.ntsteps)] 
        self.drop = nn.Dropout(dropout)

    def forward(self, input, init_hiddens):
        X = input
        time_steps = X.size(0)
        #assert time_steps == self.ntsteps # TODO: doesn't work because at end of epoch
        batch_size = X.size(1)
        # NOTATION: [] is an array
        output = [] # nested_list (ntsteps, [bs,nh])
        h_t, c_t = init_hiddens # init_hiddens: (2, [nlayers,bs,nh]); h_t: (<nlayers, [bs, nh])

        for tstep in range(time_steps):
            h_tm1, c_tm1 = h_t, c_t
            h_t = []
            c_t = []
            for layer in range(self.nlayers): 
                if layer == 0:
                    x_input = X[tstep]
                else:
                    x_input = self.drop(h_t[-1]) # take the hiddens from the layer below as input and apply dropout
                hh, cc = h_tm1[layer], c_tm1[layer]
                inp = torch.cat( (x_input, hh), 1 )
                gates = self.i2h[tstep][layer](inp)
                # TODO
                hh, cc = LSTMCell(gates, cc)
                h_t.append(hh)
                c_t.append(cc)
            output.append(torch.unsqueeze(hh,0)) # add a singleton time-step dimension
        # (seq_len, batch, hidden_size * num_directions): tensor containing the output features (h_k)
        outputs = torch.cat(output, 0)
        # (final) hiddens = (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for k=seq_len.
        hiddens = (torch.cat([torch.unsqueeze(hh,0) for hh in h_t], 0),
                   torch.cat([torch.unsqueeze(cc,0) for cc in c_t], 0))
        return outputs, hiddens




class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)

        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        #print output
        # output = TBN
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
