import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence


def dynamicRNN(rnnModel,
               seqInput,
               seqLens,
               initialState=None,
               returnStates=False):
    '''
    Inputs:
        rnnModel     : Any torch.nn RNN model
        seqInput     : (batchSize, maxSequenceLength, embedSize)
                        Input sequence tensor (padded) for RNN model
        seqLens      : batchSize length torch.LongTensor or numpy array
        initialState : Initial (hidden, cell) states of RNN

    Output:
        A single tensor of shape (batchSize, rnnHiddenSize) corresponding
        to the outputs of the RNN model at the last time step of each input
        sequence. If returnStates is True, also return a tuple of hidden
        and cell states at every layer of size (num_layers, batchSize,
        rnnHiddenSize)
    '''
    sortedLen, fwdOrder, bwdOrder = getSortedOrder(seqLens)
    sortedSeqInput = seqInput.index_select(dim=0, index=fwdOrder)
    packedSeqInput = pack_padded_sequence(
        sortedSeqInput, lengths=sortedLen, batch_first=True)

    if initialState is not None:
        hx = initialState
        sortedHx = [x.index_select(dim=1, index=fwdOrder) for x in hx]
        assert hx[0].size(0) == rnnModel.num_layers  # Matching num_layers
    else:
        hx = None

    rnnModel.flatten_parameters()
    _, (h_n, c_n) = rnnModel(packedSeqInput, hx)

    rnn_output = h_n[-1].index_select(dim=0, index=bwdOrder)

    if returnStates:
        h_n = h_n.index_select(dim=1, index=bwdOrder)
        c_n = c_n.index_select(dim=1, index=bwdOrder)
        return rnn_output, (h_n, c_n)
    else:
        return rnn_output