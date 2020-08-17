import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


class GlobalAttention(nn.Module):

    def __init__(self, dim):
        super(GlobalAttention, self).__init__()
        self.dim = dim
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)

    def forward(self, queries, memory_bank, memory_lengths):
        """
                    (BxT'xd)  (BxTxd)  B  -------->   (T'xBxd) (T'xBxT)      or
                       (Bxd)  (BxTxd)  B  -------->      (Bxd) (T'xBxT)
        """
        # one step input (Bxd)
        if queries.dim() == 2:
            one_step = True
            queries = queries.unsqueeze(1)
        else:
            one_step = False

        align = self.score(queries, memory_bank)  # BxT'xT

        T_tgt = queries.size(1)
        B, T_src, d = memory_bank.size()

        mask = sequence_mask(memory_lengths, max_len=align.size(-1))
        mask = mask.unsqueeze(1)  # Make it broadcastable: {0,1}^{B x 1 x T}
        align.masked_fill_(~mask,
                           -float('inf'))  #  align[b][t'][t>len(b)] = -inf

        align_vectors = F.softmax(align.view(B*T_tgt, T_src), -1)
        align_vectors = align_vectors.view(B, T_tgt, T_src)

        #              (B x T' x T) (B x T x d) ---------> (B x T' x d)
        c = torch.bmm(align_vectors, memory_bank)

        # TODO: attn_h is the LHS of Eq. (5) in Luong et al. (2015).
        # Implement it using c, queries, and self.linear_out.
        attn_h = torch.tanh(self.linear_out(torch.cat((c, queries), dim=2)))

        if one_step:  # Bx1xd ----> Bxd
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)
        else:  # BxT'xd ----> T'xBxd
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()

        # (T' x B x d)  (T' x B x T)  if not one_step
        #      (B x d)       (B x T)  if     one_step
        return attn_h, align_vectors

    def score(self, Q, K):
        """
               (BxT'xd) (BxTxd)  ------->  (BxT'xT)
        """
        L = torch.bmm(self.linear_in(Q), K.transpose(1,2))
        return L  # TODO: Implement using self.linear_in.
