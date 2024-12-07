import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader


class KGDataset(Dataset):
    def __init__(self, data_path):
        triples = []
        with open(data_path) as f:
            for line in f:
                head, rel, tail = line.strip().split()
                triples.append((int(head), int(rel), int(tail)))
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        head, rel, tail = self.triples[idx]
        return torch.LongTensor([head, rel, tail])


class KGEmb(nn.Module):
    def __init__(self, n_ent, n_rel, dim=10):
        """
        :param n_ent: Number of entities
        :param n_rel: Number of relations
        :param dim: Dimension of the knowledge embedding
        """
        super().__init__()
        self.ent_emb = nn.Embedding(n_ent, dim)
        self.rel_emb = nn.Embedding(n_rel, dim)

    def forward(self, x):
        head, rel, tail = x[:, 0], x[:, 1], x[:, 2]
        head_emb = self.ent_emb(head)
        rel_emb = self.rel_emb(rel)
        tail_emb = self.ent_emb(tail)
        return head_emb, rel_emb, tail_emb


class TransE:
    def __init__(self, save_path, data_path, ent_num, rel_num, dim):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dataset = KGDataset(data_path)
        torch.manual_seed(123)
        model = KGEmb(n_ent=ent_num, n_rel=rel_num, dim=dim).to(device)
        opt = torch.optim.Adam(model.parameters())
        loader = DataLoader(self.dataset, batch_size=256, shuffle=True)

        for epoch in range(200):
            print("Epoch: ", epoch)
            for step, x in tqdm(enumerate(loader), total=len(loader)):
                opt.zero_grad()
                x = x.to(device)
                h, r, t = model(x)
                loss = torch.sum(h + r - t).to(device)  # Loss
                loss.backward()
                opt.step()

        self.ent_emb = model.ent_emb.weight.detach().cpu().numpy()
        self.rel_emb = model.rel_emb.weight.detach().cpu().numpy()
        torch.save(model.state_dict(), save_path + 'model.pkl')

        self.head_ent_emb = self.ent_emb[4]
        self.tail_ent_emb = self.ent_emb[10]
