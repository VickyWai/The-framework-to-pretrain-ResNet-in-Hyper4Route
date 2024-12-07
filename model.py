import torch
from torch import nn
from subgraph_encoder import SubGraphEncoder
import torch.nn.functional as F
from kg_encorder import KGEncoder
from satellite_encoder import ImageEncoder


class CompleteModel(nn.Module):
    def __init__(self,
                 mesh_num,
                 sub_node_feature_input_dim,
                 sub_node_feature_output_dim,
                 kg_embedding_dim,
                 img_input_dim,
                 img_output_dim,
                 pretrained_weight_path,
                 ent_emb,
                 rel_emb,
                 dropout,
                 batch_size,
                 **kwargs):
        super(CompleteModel, self).__init__()
        """
        sub_node_feature_input_dim: dim of initial feature embeddings
        sub_node_feature_output_dim: dim of output feature embeddings
        kg_embedding_dim: dim of kg embeddings
        img_input_dim: size of input image
        img_output_dim: dim of output image embeddings
        pretrained_weight_path: the pretrained weights of ViT or ResNet
        batch_size: batch size of 4x4
        """

        self.d = kwargs['d']
        self.g = kwargs['g']
        self.mesh_num = mesh_num
        self.ent_emb = nn.Embedding.from_pretrained(ent_emb, freeze=True)
        self.rel_emb = nn.Embedding.from_pretrained(rel_emb, freeze=True)
        self.batch_size = batch_size
        self.sub_node_feature_output_dim = sub_node_feature_output_dim

        self.transport_sub_graph = SubGraphEncoder(sub_node_feature_input_dim, sub_node_feature_output_dim,
                                                   dropout, batch_size, mesh_num)

        self.roadnet_sub_graph = SubGraphEncoder(sub_node_feature_input_dim, sub_node_feature_output_dim,
                                                 dropout, batch_size, mesh_num)

        self.function_sub_graph = SubGraphEncoder(sub_node_feature_input_dim, sub_node_feature_output_dim,
                                                  dropout, batch_size, mesh_num)

        self.city_sub_graph = SubGraphEncoder(sub_node_feature_input_dim, sub_node_feature_output_dim,
                                              dropout, batch_size, mesh_num)

        self.degree_sub_graph = SubGraphEncoder(sub_node_feature_input_dim, sub_node_feature_output_dim,
                                                dropout, batch_size, mesh_num)
        self.graph_projector = nn.Sequential(
            nn.Linear(sub_node_feature_output_dim, sub_node_feature_output_dim, bias=False),
            nn.ReLU(),
            nn.Linear(sub_node_feature_output_dim, sub_node_feature_output_dim, bias=False)
        )
        torch.nn.init.xavier_normal_(self.graph_projector[0].weight.data)
        torch.nn.init.xavier_normal_(self.graph_projector[2].weight.data)

        self.kg_whole_graph = nn.ModuleList()
        self.kg_whole_graph.append(KGEncoder(kg_embedding_dim, sub_node_feature_output_dim))
        self.kg_whole_graph.append(KGEncoder(sub_node_feature_output_dim, sub_node_feature_output_dim))
        self.kg_projector = nn.Sequential(
            nn.Linear(sub_node_feature_output_dim, sub_node_feature_output_dim, bias=False),
            nn.ReLU(),
            nn.Linear(sub_node_feature_output_dim, sub_node_feature_output_dim, bias=False),
        )
        torch.nn.init.xavier_normal_(self.kg_projector[0].weight.data)
        torch.nn.init.xavier_normal_(self.kg_projector[2].weight.data)

        self.img_encoder = ImageEncoder(img_input_dim, img_output_dim,
                                                 pretrained_weight_path, batch_size, self.mesh_num)
        mlp_pretrain_sv = torch.load(pretrained_weight_path)
        self.img_projector = nn.Sequential(
            nn.Linear(img_output_dim, img_output_dim, bias=False),
            nn.ReLU(),
            nn.Linear(img_output_dim, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, sub_node_feature_output_dim, bias=False),
            nn.ReLU()
        )
        self.img_projector[0].weight.data = mlp_pretrain_sv['projector.0.weight']
        self.img_projector[2].weight.data = mlp_pretrain_sv['projector.2.weight']

        self.dropout = nn.Dropout(dropout)

        self.graph_fusion = nn.Sequential(
            nn.Linear(self.sub_node_feature_output_dim * 6, self.sub_node_feature_output_dim * 3),
            nn.ReLU(),
            nn.Linear(self.sub_node_feature_output_dim * 3, self.sub_node_feature_output_dim),
            nn.ReLU()
        )

    def sub_graph_inner_loss(self, transport, roadnet, function, city, degree):
        loss = 0
        for a, b in [[transport, roadnet], [transport, function], [transport, city],
                     [transport, degree], [roadnet, function], [roadnet, city],
                     [roadnet, degree], [function, city], [function, degree], [city, degree]]:
            score = torch.einsum('ai, bi->ab', a, b)

            score_1 = F.softmax(score, dim=1)
            diag_1 = torch.diag(score_1)
            loss = loss - torch.log(diag_1 + 1e-10).sum()

            score_2 = F.softmax(score, dim=0)
            diag_2 = torch.diag(score_2)
            loss = loss - torch.log(diag_2 + 1e-10).sum()
        return loss

    def sub_graph_kg_loss(self, transport, roadnet, function, city, degree, kg_mesh_features):
        loss = 0
        for a in [transport, roadnet, function, city, degree]:
            score = torch.einsum('ai, bi->ab', a, kg_mesh_features)

            score_1 = F.softmax(score, dim=1)
            diag_1 = torch.diag(score_1)
            loss = loss - torch.log(diag_1 + 1e-10).sum()

            score_2 = F.softmax(score, dim=0)
            diag_2 = torch.diag(score_2)
            loss = loss - torch.log(diag_2 + 1e-10).sum()
        return loss

    def global_kg_img_loss(self, kg, img):
        loss = 0
        score = torch.einsum('ai, bi->ab', kg, img)

        score_1 = F.softmax(score, dim=1)
        diag_1 = torch.diag(score_1)
        loss = loss - torch.log(diag_1 + 1e-10).sum()

        score_2 = F.softmax(score, dim=0)
        diag_2 = torch.diag(score_2)
        loss = loss - torch.log(diag_2 + 1e-10).sum()
        return loss

    def local_kg_img_loss(self, kg, img):
        loss = 0
        score = torch.einsum('ai, bi->ab', kg, img)

        score_1 = F.softmax(score, dim=1)
        diag_1 = torch.diag(score_1)
        loss = loss - torch.log(diag_1 + 1e-10).sum()

        score_2 = F.softmax(score, dim=0)
        diag_2 = torch.diag(score_2)
        loss = loss - torch.log(diag_2 + 1e-10).sum()
        return loss

    def local_kg_img_cat(self, transport, roadnet, function, city, degree, kg):
        local_fused_graph = torch.cat((transport, roadnet, function, city, degree, kg), dim=1)
        return local_fused_graph

    def forward(self, node_feature_group,  # b*16, 10 ent_emb初始表示
                transport_embedding_group,  # b*16, 10
                roadnet_embedding_group,  # b*16, 10
                function_embedding_group,
                city_embedding_group,
                degree_embedding_group,
                sudoku_image_list,  # b*16, 3, 224, 224
                region_index,  # 1, b
                mesh_sudoku_id):  # b. 16
        transport = self.graph_projector(self.transport_sub_graph(node_feature_group, transport_embedding_group))
        roadnet = self.graph_projector(self.roadnet_sub_graph(node_feature_group, roadnet_embedding_group))
        function = self.graph_projector(self.function_sub_graph(node_feature_group, function_embedding_group))
        city = self.graph_projector(self.city_sub_graph(node_feature_group, city_embedding_group))
        degree = self.graph_projector(self.degree_sub_graph(node_feature_group, degree_embedding_group))
        loss_sub_inner = self.sub_graph_inner_loss(transport, roadnet, function, city, degree)

        ent_feature = self.ent_emb.weight
        rel_feature = self.rel_emb.weight
        for layer in self.kg_whole_graph:
            ent_feature, rel_feature = layer(self.g, ent_feature, rel_feature)
            ent_feature = self.dropout(ent_feature)
        ent_feature = self.kg_projector(ent_feature)
        kg_mesh_group = ent_feature[mesh_sudoku_id.cpu().numpy().tolist(), :].\
            reshape((self.batch_size * self.mesh_num, -1))
        loss_sub_kg = self.sub_graph_kg_loss(transport, roadnet, function, city, degree, kg_mesh_group)

        img_emb_group = self.img_projector(self.img_encoder(sudoku_image_list))  # [b * mesh_num, 10]
        local_fused_graph = self.graph_fusion(self.local_kg_img_cat
                                              (transport, roadnet, function, city, degree, kg_mesh_group))
        loss_local_kg_img = self.local_kg_img_loss(local_fused_graph, img_emb_group)

        img_emb_group_reshape = img_emb_group.reshape((self.batch_size, self.mesh_num, -1))
        region_img_emb = torch.mean(img_emb_group_reshape, dim=1)
        kg_region_features = ent_feature[region_index, :]
        loss_global_kg_img = self.global_kg_img_loss(kg_region_features, region_img_emb)

        loss_total = loss_sub_inner.requires_grad_(True) \
                     + loss_sub_kg.requires_grad_(True) \
                     + loss_local_kg_img.requires_grad_(True) \
                     + loss_global_kg_img.requires_grad_(True)

        return loss_total.requires_grad_(True), loss_sub_inner, loss_sub_kg, loss_local_kg_img, loss_global_kg_img
