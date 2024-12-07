import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SubGraphEncoder(nn.Module):
    def __init__(self,
                 sub_node_feature_input_dim,
                 sub_node_feature_output_dim,
                 dropout,
                 batch_size,
                 mesh_num,
                 symmetric=True):
        super(SubGraphEncoder, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_dim = sub_node_feature_input_dim
        self.output_dim = sub_node_feature_output_dim
        self.symmetric = symmetric
        self.mesh_num = mesh_num
        self.batch_size = batch_size

        self.dad = torch.zeros((self.batch_size, self.mesh_num, self.mesh_num))

        self.gcn1 = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.gcn2 = nn.Linear(self.output_dim, self.output_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def adj_cal(self, embedding):
        embedding_block = embedding.reshape((self.batch_size, self.mesh_num, -1))
        adj = torch.mm(embedding_block[0], embedding_block[0].T)
        adj = torch.clamp(adj, min=0)
        for i in range(1, self.batch_size):
            adj_block = torch.mm(embedding_block[i], embedding_block[i].T)
            adj_block = torch.clamp(adj_block, min=0)
            adj = torch.block_diag(adj, adj_block)
        A = adj.to(device) + torch.eye(self.batch_size * self.mesh_num).to(device)
        d = A.sum(1)
        if self.symmetric:
            D = torch.diag(torch.pow(d, -0.5))
            return D.mm(A).mm(D)
        else:
            D = torch.diag(torch.pow(d, -1))
            return D.mm(A)

    def forward(self, node_feature_group, embedding_group):
        dad = self.adj_cal(embedding_group)
        y = F.relu(self.gcn1(dad.mm(node_feature_group)))
        y = F.relu(self.gcn2(dad.mm(y)))
        output = self.dropout(y)
        return output
