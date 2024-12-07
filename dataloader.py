import json
import os
from tqdm import tqdm
import dgl
from PIL import Image
import torchvision
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from transe import TransE


class DataPrepare:
    def __init__(self, img_fold_path, embedding_file_path, kg_file_path):
        self.img_list = os.listdir(img_fold_path)
        self.img_mesh_list = [x.replace(".jpg", '') for x in self.img_list]  # str
        with open(embedding_file_path, 'r') as jf:
            self.embedding_dict = json.load(jf)
        self.attribute_mesh_list = list(self.embedding_dict.keys())  # str
        self.available_mesh_list = list(set(self.attribute_mesh_list) & set(self.img_mesh_list))

        self.city_kg_file_path = kg_file_path

        self.ents, self.rels, self.ent2id, self.rel2id, \
            self.kg_data, self.mesh_kg_index, self.region_kg_index = self.load_kg()
        """
        ents: list str
        rel: list str
        ent2id: dict{str: int}
        rel2id: dict{str: int}
        kg_data: list int [[ent-rel-ent], [ent-rel-ent], [ent-rel-ent]...]
        mesh_kg_index: list int [0, 1, 2, 3, ...]
        region_kg_index: list int [0, 1, 2, 3, ...]
        """
        self.num_ent, self.num_rel = len(self.ent2id), len(self.rel2id) // 2
        self.id2ent = {v: k for k, v in self.ent2id.items()}  # id: int, ent: str
        rels = [x[1] for x in self.kg_data]  # int
        src = [x[0] for x in self.kg_data]  # int
        dst = [x[2] for x in self.kg_data]  # int
        self.g = dgl.graph((src, dst), num_nodes=self.num_ent)
        self.g.edata['etype'] = torch.Tensor(rels).long()

        in_edges_mask = [True] * (self.g.num_edges() // 2) + [False] * (self.g.num_edges() // 2)
        out_edges_mask = [False] * (self.g.num_edges() // 2) + [True] * (self.g.num_edges() // 2)
        self.g.edata['in_edges_mask'] = torch.Tensor(in_edges_mask)
        self.g.edata['out_edges_mask'] = torch.Tensor(out_edges_mask)

    def load_kg(self):
        facts_str = []
        print('loading knowledge graph...')
        with open(self.city_kg_file_path, 'r') as f:
            for line in tqdm(f.readlines()):
                x = line.strip().split('\t')
                x = x[0].split()
                facts_str.append([x[0], x[1], x[2]])
        origin_rels = sorted(list(set([x[1] for x in facts_str])))
        all_rels = sorted(origin_rels + [x + '_rev' for x in origin_rels])
        all_ents = sorted(list(set([x[0] for x in facts_str] + [x[2] for x in facts_str])))

        mesh_ents = [x for x in all_ents if '_' not in x]  # str
        other_ents = [x for x in all_ents if '_' in x and 'region' not in x]
        region_ents = [x for x in all_ents if "region" in x]

        mesh_ents = sorted(mesh_ents, key=lambda y: int(y))  # str
        other_ents = sorted(other_ents)
        region_ents = [x.replace('region_', '') for x in region_ents]
        region_ents = sorted(region_ents, key=lambda y: int(y))  # str
        region_ents = ['region_' + x for x in region_ents]

        ents = mesh_ents + region_ents + other_ents  # str

        ent2id, rel2id = dict([(x, i) for i, x in enumerate(ents)]), dict([(x, i) for i, x in enumerate(all_rels)])
        kg_data = [[ent2id[x[0]], rel2id[x[1]], ent2id[x[2]]] for x in facts_str] + \
                  [[ent2id[x[2]], rel2id[x[1] + '_rev'], ent2id[x[2]]] for x in facts_str]  # int
        mesh_kg_index = [ent2id[x] for x in mesh_ents]  # int
        region_kg_index = [ent2id[x] for x in region_ents]  # int
        return ents, all_rels, ent2id, rel2id, kg_data, mesh_kg_index, region_kg_index


class CityDataset(Dataset):
    def __init__(self, region_kg_index, mesh_kg_index, ent2id,
                 id2ent, img_fold_path, embedding_fold_path, img_input_size):
        self.region_kg_index = region_kg_index
        self.mesh_kg_index = mesh_kg_index
        self.id2ent = id2ent
        self.ent2id = ent2id
        self.city_img_path = img_fold_path
        self.city_embedding_path = embedding_fold_path
        with open(self.city_embedding_path, 'r') as jf:
            self.sudoku_rep_dict = json.load(jf)
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((img_input_size, img_input_size)),
                torchvision.transforms.ToTensor(),
            ]
        )
        self.transform = self.transform
        if len(self.region_kg_index) != len(self.mesh_kg_index):
            print('Wrong model !!!')
            exit(0)

    def __len__(self):
        return len(self.mesh_kg_index)

    def __getitem__(self, index):
        # 第index笔数据
        mesh_index = self.mesh_kg_index[index]
        region_index = self.region_kg_index[index]
        sudoku_image_list = []
        transport_graph = []
        roadnet_graph = []
        function_graph = []
        city_graph = []
        degree_graph = []
        mesh_ents = self.id2ent[mesh_index]
        mesh_sudoku_list = sorted(list(self.sudoku_rep_dict[mesh_ents].keys()), key=lambda y: int(y))
        mesh_sudoku_id = [self.ent2id[x] for x in mesh_sudoku_list]

        for sudoku_mesh in mesh_sudoku_list:
            sudoku_mesh_img_path = os.path.join(self.city_img_path, sudoku_mesh + '.jpg')
            sudoku_img = Image.open(sudoku_mesh_img_path)
            if self.transform:
                sudoku_img = self.transform(sudoku_img)
            sudoku_image_list.append(sudoku_img)

            sudoku_mesh_embedding_dict = self.sudoku_rep_dict[mesh_ents][sudoku_mesh]
            transport_graph.append(torch.tensor(sudoku_mesh_embedding_dict['transport_embedding']))
            roadnet_graph.append(torch.tensor(sudoku_mesh_embedding_dict['roadnet_embedding']))
            function_graph.append(torch.tensor(sudoku_mesh_embedding_dict['function_embedding']))
            city_graph.append(torch.tensor(sudoku_mesh_embedding_dict['city_embedding']))
            degree_graph.append(torch.tensor(sudoku_mesh_embedding_dict['degree_embedding']))

        sudoku_image_list = torch.stack(sudoku_image_list, dim=0)  # 16*3*224*224
        transport_graph = torch.stack(transport_graph, dim=0)  # 16*10
        roadnet_graph = torch.stack(roadnet_graph, dim=0)  # 16*10
        function_graph = torch.stack(function_graph, dim=0)  # 16*10
        city_graph = torch.stack(city_graph, dim=0)  # 16*10
        degree_graph = torch.stack(degree_graph, dim=0)  # 16*10
        mesh_sudoku_id = torch.Tensor(mesh_sudoku_id)  # 1, 16

        return transport_graph, roadnet_graph, function_graph, city_graph, degree_graph, \
            sudoku_image_list, region_index, mesh_sudoku_id, mesh_index


class KGEmbedding:
    def __init__(self,
                 output_dim,
                 ent2id, rel2id, kg,
                 path,
                 transe_model_save_path,
                 load_pretrained_emb=True,
                 write_file=False,
                 embedding_train=False):
        self.entity2id = ent2id  # dict
        self.relation2id = rel2id  # dict
        self.triple = kg  # list(int): [[ent-rel-ent], [ent-rel-ent], [ent-rel-ent]...]

        self.entity_num = len(self.entity2id)
        self.relation_num = len(self.relation2id)

        self.model_path = transe_model_save_path
        self.id2entity_path = os.path.join(path, "id2entity.txt")
        self.id2relation_path = os.path.join(path, "id2relation.txt")
        self.triple_path = os.path.join(path, "triple.txt")

        self.ent_emb_path = os.path.join(path, "ent_emb.npy")
        self.rel_emb_path = os.path.join(path, "rel_emb.npy")

        if write_file:
            self.write_openke_prepare_flies()
        if embedding_train:
            KGE = TransE(self.model_path, self.triple_path, self.entity_num, self.relation_num, output_dim)
            self.ent_emb = KGE.ent_emb
            self.rel_emb = KGE.rel_emb
            np.save(self.ent_emb_path, self.ent_emb)
            np.save(self.rel_emb_path, self.rel_emb)
        if load_pretrained_emb:
            self.load_pretrained_kg_ent_rel_emb()

    def write_openke_prepare_flies(self):
        ent_str = []
        ent_str = ent_str + [str(value) + '\t' + str(key) + '\n'
                             for key, value in sorted(self.entity2id.items(), key=lambda item: item[1])]
        with open(self.id2entity_path, 'w') as tf:
            tf.writelines(ent_str)

        rel_str = []
        rel_str = rel_str + [str(value) + '\t' + str(key) + '\n'
                             for key, value in sorted(self.relation2id.items(), key=lambda item: item[1])]
        with open(self.id2relation_path, 'w') as tf:
            tf.writelines(rel_str)

        triple_str = []
        triple_str = triple_str + [str(x[0]) + '\t' + str(x[1]) + '\t' + str(x[2]) + '\n' for x in self.triple]
        with open(self.triple_path, 'w') as tf:
            tf.writelines(triple_str)

    def load_pretrained_kg_ent_rel_emb(self):
        self.ent_emb = np.load(self.ent_emb_path)
        self.rel_emb = np.load(self.rel_emb_path)

