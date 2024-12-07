import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
import torch
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import argparse
from dataloader import DataPrepare, CityDataset, KGEmbedding
from utils import in_out_norm
from model import CompleteModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def trainer(args, model, ent_emb, train_loader, optimizer, epoch, id2ent, ent2id, embedding_dict):
    loss_epoch, loss1_epoch, loss2_epoch, loss3_epoch, loss4_epoch = [], [], [], [], []
    model.train()
    for step, batch in enumerate(tqdm(train_loader, total=len(train_loader))):
        optimizer.zero_grad()
        transport_graph, roadnet_graph, function_graph, city_graph, \
            degree_graph, sudoku_image_list, region_index, mesh_sudoku_id, mesh_index = batch

        transport_graph = transport_graph.reshape((args.batch_size * args.mesh_num, -1))
        roadnet_graph = roadnet_graph.reshape((args.batch_size * args.mesh_num, -1))
        function_graph = function_graph.reshape((args.batch_size * args.mesh_num, -1))
        city_graph = city_graph.reshape((args.batch_size * args.mesh_num, -1))
        degree_graph = degree_graph.reshape((args.batch_size * args.mesh_num, -1))
        sudoku_image_list = sudoku_image_list.reshape(
            (args.batch_size * args.mesh_num, 3, args.img_input_dim, args.img_input_dim)
        )
        mesh_sudoku_id = mesh_sudoku_id.reshape((args.batch_size, -1))

        mesh_ent = [id2ent[x] for x in mesh_index.numpy()]
        sudo_mesh_all = [list(embedding_dict[x].keys()) for x in mesh_ent]
        mesh_emb = torch.zeros((args.batch_size, args.mesh_num, args.kg_embedding_dim))

        for i, j in enumerate(sudo_mesh_all):
            j = [ent2id[x] for x in j]
            mesh_emb[i] = torch.tensor(ent_emb[j, :])
        node_feature = mesh_emb.reshape((args.batch_size * args.mesh_num, args.kg_embedding_dim)).to(args.device)

        loss, loss_sub_inner, loss_sub_kg, loss_local_kg_img, loss_global_kg_img = \
            model(node_feature, transport_graph.to(args.device), roadnet_graph.to(args.device),
                  function_graph.to(args.device), city_graph.to(args.device), degree_graph.to(args.device),
                  sudoku_image_list.to(args.device), region_index.to(args.device), mesh_sudoku_id.to(args.device))

        loss.backward()
        optimizer.step()
        loss_epoch.append(loss.item())
        loss1_epoch.append(loss_sub_inner.item())
        loss2_epoch.append(loss_sub_kg.item())
        loss3_epoch.append(loss_local_kg_img.item())
        loss4_epoch.append(loss_global_kg_img.item())
        if step % 20 == 0:
            print(f"TrainStep [{step}/{len(train_loader)}]\t train_loss: {loss.item()}")
        args.global_step += 1
    print(f"TrainEpoch [{epoch}/{args.epochs}\t train_loss_epoch:{np.mean(loss_epoch)}")
    return np.mean(loss_epoch), np.mean(loss1_epoch), np.mean(loss2_epoch), np.mean(loss3_epoch), np.mean(loss4_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ours")
    parser.add_argument("--city", type=str, default="Shanghai")
    parser.add_argument("--mesh_num", type=int, default=16)
    parser.add_argument("--embedding_file_path", type=str, default="")
    parser.add_argument("--img_fold_path", type=str, default="")
    parser.add_argument("--kg_file_path", type=str, default="")
    parser.add_argument("--kg_embedding_path", type=str, default="")
    parser.add_argument("--TransE_model_path", type=str, default="")
    parser.add_argument("--kg_embedding_dim", type=int, default=10, help="The dim of the node/edge in KG")
    parser.add_argument("--gcn_dropout", type=float, default=0.1)
    parser.add_argument("--load_pretrained_emb", type=bool, default=True, help="load the TransE embedding of KG")
    parser.add_argument("--write_train_file", type=bool, default=True, help="The three files for training TransE")
    parser.add_argument("--TransE_train", type=bool, default=True, help="Train KG embeddings or not")
    parser.add_argument("--sub_node_feature_input_dim", type=int, default=10, help="")
    parser.add_argument("--sub_node_feature_output_dim", type=int, default=10, help="")
    parser.add_argument("--img_input_dim", type=int, default=224)
    parser.add_argument("--img_output_dim", type=int, default=512)
    parser.add_argument("--pretrained_weight_path", type=str, default="")
    parser.add_argument("--start_epoch", type=int, default=0, nargs="?", help="Start epoch")
    parser.add_argument("--current_epoch", type=int, default=0, nargs="?", help="Current epoch")
    parser.add_argument("--global_step", type=int, default=0, nargs="?", help="global_step")
    parser.add_argument("--epochs", type=int, default=150, nargs="?", help="Epochs")
    parser.add_argument("--batch_size", type=int, default=4, nargs="?", help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0003, nargs="?", help="Learning rate.")
    parser.add_argument("--seed", type=int, default=2023, nargs="?", help="")
    parser.add_argument("--model_path", type=str, default="")

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.embedding_file_path = ".\\data\\CitySudoku\\4_4.json"
    args.img_fold_path = ".\\data\\CitySatellite\\" + args.city
    args.kg_file_path = ".\\data\\CityKnowledgeGraph\\" + args.city + '.txt'
    args.kg_embedding_path = ".\\CityKGEmbedding\\" + args.city
    args.TransE_model_path = ".\\CityKGEmbedding\\" + args.city + "\\model\\"
    # args.pretrained_weight_path = ".\\ViT-B_16\\base_p16_224_backbone.pth"  # ImageEncoder: ViT
    args.pretrained_weight_path = ".\\model_weights\\checkpoint_100.tar"  # ImageEncoder: ResNet
    args.model_path = os.path.join(os.getcwd(), 'model_save')

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(False)

    d = DataPrepare(args.img_fold_path, args.embedding_file_path, args.kg_file_path)
    '''
    img_fold_path: satellite imagery fold
    embedding_file_path: feature embedding files of batches
    kg_file_path: urban knowledge graph
    '''
    kg_init_embedding = KGEmbedding(args.kg_embedding_dim, d.ent2id, d.rel2id, d.kg_data, args.kg_embedding_path,
                                    args.TransE_model_path, load_pretrained_emb=args.load_pretrained_emb,
                                    write_file=args.write_train_file, embedding_train=args.TransE_train)  # train KGEmb
    '''
    kg_embedding_dim: output dim of TransE
    kg_embedding_path: kg embedding saving path
    TransE_model_path: TranE model saving path
    '''
    ent_emb = kg_init_embedding.ent_emb
    rel_emb = kg_init_embedding.rel_emb

    with open(args.embedding_file_path, 'r') as jf:
        embedding_dict = json.load(jf)
    sudoku_region_list = list(embedding_dict.keys())  # region label list
    img_list = os.listdir(args.img_fold_path)
    img_mesh_list = [x.replace('.jpg', '') for x in img_list]
    available_region_list = list(set(sudoku_region_list) & set(img_mesh_list))
    embedding_dict_available = {k: embedding_dict[k] for k in available_region_list}

    mesh_kg_ents = sorted(list(available_region_list), key=lambda y: int(y))
    region_kg_ents = ['region_' + x for x in mesh_kg_ents]
    '''
    mesh_kg_ents: sort the labels of available grids
    region_kg_ents: sort the labels of available regions
    During training, each batch of input units is a region and its set of retrievable grids.
    '''
    mesh_kg_index = [d.ent2id[x] for x in mesh_kg_ents]
    region_kg_index = [d.ent2id[x] for x in region_kg_ents]
    '''
    mesh_kg_index: labels of sorted available grids
    region_kg_index: labels of available regions
    '''
    train_dataset = CityDataset(region_kg_index, mesh_kg_index, d.ent2id, d.id2ent,
                                args.img_fold_path, args.embedding_file_path, args.img_input_dim)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    print('len of train_dataset:', len(train_dataset))

    g = in_out_norm(d.g.to(args.device)).to(args.device)
    kwargs = {'d': d, 'g': g}
    ent_emb = torch.tensor(ent_emb).to(args.device)
    rel_emb = torch.tensor(rel_emb).to(args.device)

    model = CompleteModel(args.mesh_num,
                          args.sub_node_feature_input_dim,
                          args.sub_node_feature_output_dim,
                          args.kg_embedding_dim,
                          args.img_input_dim,
                          args.img_output_dim,
                          args.pretrained_weight_path,
                          ent_emb,
                          rel_emb,
                          args.gcn_dropout,
                          args.batch_size,
                          **kwargs)
    model = model.to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    loss0, loss1, loss2, loss3, loss4 = [], [], [], [], []
    for epoch in range(args.start_epoch, args.epochs):
        loss_epoch, loss1_epoch, loss2_epoch, loss3_epoch, loss4_epoch = \
            trainer(args, model, ent_emb, train_loader, opt, epoch, d.id2ent, d.ent2id, embedding_dict_available)
        if epoch in range(0, args.epochs, 20):
            out = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.current_epoch))
            torch.save(model.state_dict(), out)
        args.current_epoch += 1
        loss0.append(loss_epoch)
        loss1.append(loss1_epoch)
        loss2.append(loss2_epoch)
        loss3.append(loss3_epoch)
        loss4.append(loss4_epoch)
    out = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.current_epoch))
    torch.save(model.state_dict(), out)

    np.savez('loss_total', loss0)
    np.savez('loss_sub_inner', loss1)
    np.savez('loss_sub_kg', loss2)
    np.savez('loss_local_kg_img', loss3)
    np.savez('loss_global_kg_img', loss4)
