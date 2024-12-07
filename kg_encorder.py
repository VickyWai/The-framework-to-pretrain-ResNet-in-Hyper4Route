import torch
import torch.nn as nn
import dgl.function as fn
from utils import ccorr


def rotate(h, r):
    d = h.shape[-1]
    h_re, h_im = torch.split(h, d // 2, -1)
    r_re, r_im = torch.split(r, d // 2, -1)
    return torch.cat([h_re * r_re - h_im * r_im, h_re * r_im + h_im * r_re], dim=-1)


class KGEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 comp_fn='rotate',
                 batchnorm=True,
                 dropout=0.1):
        super(KGEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.comp_fn = comp_fn
        self.activation = torch.tanh
        self.batchnorm = batchnorm

        self.dropout = nn.Dropout(dropout)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(output_dim)

        self.W_O = nn.Linear(self.input_dim, self.output_dim)
        self.W_I = nn.Linear(self.input_dim, self.output_dim)
        self.W_S = nn.Linear(self.input_dim, self.output_dim)

        self.W_R = nn.Linear(self.input_dim, self.output_dim)

        self.loop_rel = nn.Parameter(torch.Tensor(1, self.input_dim))
        nn.init.xavier_normal_(self.loop_rel)

    def forward(self, g, ent_input_feature, rel_input_feature):
        with g.local_scope():
            g.srcdata['h'] = ent_input_feature
            rel_input_feature = torch.cat((rel_input_feature, self.loop_rel), 0)
            g.edata['h'] = rel_input_feature[g.edata['etype']] * g.edata['norm']

            if self.comp_fn == 'sub':
                g.apply_edges(fn.u_sub_e('h', 'h', out='comp_h'))
            elif self.comp_fn == 'mul':
                g.apply_edges(fn.u_mul_e('h', 'h', out='comp_h'))
            elif self.comp_fn == 'ccorr':
                g.apply_edges(lambda edges: {'comp_h': ccorr(edges.src['h'], edges.data['h'])})
            elif self.comp_fn == 'rotate':
                g.apply_edges(lambda edges: {'comp_h': rotate(edges.src['h'], edges.data['h'])})
            else:
                raise Exception('Only supports sub, mul, and ccorr')

            comp_h = g.edata['comp_h']

            in_edges_idx = torch.nonzero(g.edata['in_edges_mask'], as_tuple=False).squeeze()
            out_edges_idx = torch.nonzero(g.edata['out_edges_mask'], as_tuple=False).squeeze()

            comp_h_O = self.W_O(comp_h[out_edges_idx])
            comp_h_I = self.W_I(comp_h[in_edges_idx])

            new_comp_h = torch.zeros(comp_h.shape[0], self.output_dim).to(comp_h.device)
            new_comp_h[out_edges_idx] = comp_h_O
            new_comp_h[in_edges_idx] = comp_h_I

            g.edata['new_comp_h'] = new_comp_h

            g.update_all(fn.copy_e('new_comp_h', 'm'), fn.sum('m', 'comp_edge'))

            if self.comp_fn == 'sub':
                comp_h_s = ent_input_feature - rel_input_feature[-1]
            elif self.comp_fn == 'mul':
                comp_h_s = ent_input_feature * rel_input_feature[-1]
            elif self.comp_fn == 'ccorr':
                comp_h_s = ccorr(ent_input_feature, rel_input_feature[-1])
            elif self.comp_fn == 'rotate':
                comp_h_s = rotate(ent_input_feature, rel_input_feature[-1])
            else:
                raise Exception('Only supports sub, mul, and ccorr')

            ent_out_feature = (self.W_S(comp_h_s) + self.dropout(g.ndata['comp_edge'])) * (1 / 3)

            rel_out_feature = self.W_R(rel_input_feature)

            if self.batchnorm:
                ent_out_feature = self.bn(ent_out_feature)

            if self.activation is not None:
                ent_out_feature = self.activation(ent_out_feature)

        return ent_out_feature, rel_out_feature[:-1]
