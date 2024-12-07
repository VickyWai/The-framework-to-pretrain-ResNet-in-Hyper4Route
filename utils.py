import torch as th


def ccorr(a, b):
    return th.fft.irfftn(th.conj(th.fft.rfftn(a, (-1))) * th.fft.rfftn(b, (-1)), (-1))


def in_out_norm(graph):
    src, dst, EID = graph.edges(form='all')
    graph.edata['norm'] = th.ones(EID.shape[0]).to(graph.device)

    in_edges_idx = th.nonzero(graph.edata['in_edges_mask'], as_tuple=False).squeeze()
    out_edges_idx = th.nonzero(graph.edata['out_edges_mask'], as_tuple=False).squeeze()

    for idx in [in_edges_idx, out_edges_idx]:
        u, v = src[idx], dst[idx]
        deg = th.zeros(graph.num_nodes()).to(graph.device)
        n_idx, inverse_index, count = th.unique(v, return_inverse=True, return_counts=True)
        deg[n_idx] = count.float()
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[u] * deg_inv[v]
        graph.edata['norm'][idx] = norm
    graph.edata['norm'] = graph.edata['norm'].unsqueeze(1)

    return graph
