import torch
from torch import nn


class CouEncoder(nn.Module):
    def __init__(self, cou_in_dim, cou_out_dim,
                 x_in_dim, x_out_dim, done_node_num,
                 d_h, bt,
                 batch_size=64, time_step=12):
        """
        :param cou_in_dim: Dimension of the courier's raw (input) feature vector.
        :param cou_out_dim: Dimension of the courier's encoded (output) feature vector.
        :param x_in_dim: Original dimension of the finished package input into the LSTM.
        :param x_out_dim: Expanded dimension.
        :param d_h: Dimension of hidden states (size of the courier profile).
        :param bt: Batch size * time steps (B * T)
        """
        super(CouEncoder, self).__init__()
        self.cou_emb = nn.ModuleList([nn.Linear(cou_in_dim, cou_out_dim),
                                      nn.BatchNorm1d(num_features=1, track_running_stats=False),
                                      nn.LeakyReLU()])

        self.behavior_lstm_emb = nn.Linear(x_in_dim, x_in_dim)
        self.behavior_lstm = BiLSTM(x_in_dim * done_node_num, d_h, time_step)

        self.behavior_emb = nn.ModuleList([nn.Linear(1, x_out_dim),
                                           nn.BatchNorm2d(num_features=3, track_running_stats=False),
                                           nn.LeakyReLU()])

        self.kernel_M = KernelM(done_node_num, x_in_dim, d_h)

        self.BT = bt
        self.batch_size = batch_size
        self.time_step = time_step
        self.d_past_num = done_node_num

    def forward(self, cou_fea, past_x):
        """
        :param cou_fea: B, 1, 8
        :param past_x: B*T, 3, 9
        :return: BT, 8, 8
        """
        cou_emb = cou_fea
        past_emb = past_x.unsqueeze(3)  # B*T, 3, 9, 1
        past_lstm = past_x.reshape(self.batch_size, self.time_step, -1)  # B, T, 3*9

        for layer in range(len(self.cou_emb)):
            cou_emb = self.cou_emb[layer](cou_emb)
            # B, 1, 8
        cou_emb = cou_emb * 10
        cou_temp = torch.repeat_interleave(cou_emb.unsqueeze(1), repeats=past_x.size(1), dim=1)  # B, 3, 1, 8
        cou = torch.repeat_interleave(cou_temp, repeats=self.time_step, dim=0)  # BT, 3, 1, 8

        for layer in range(len(self.behavior_emb)):
            past_emb = self.behavior_emb[layer](past_emb)  # B*T, 3, 9, 8
        beh_cou = torch.mul(past_emb, cou) * 10  # B*T, 3, 9, 8

        beh_lstm_in = self.behavior_lstm_emb(past_x)  # B*T, 3, 9
        beh_lstm_emb = self.behavior_lstm(beh_lstm_in.reshape(self.batch_size, self.time_step, -1)) * 10
        # [B, T, 8]
        beh_lstm = torch.repeat_interleave(beh_lstm_emb.unsqueeze(2).reshape(self.BT, 1, -1).unsqueeze(1),  # BT,1,1,8
                                           repeats=past_x.size(1), dim=1)  # B*T, 3, 1, 8

        beh_cou_final = torch.mul(beh_cou, beh_lstm)  # B*T, 3, 9, 8

        out = self.kernel_M(beh_cou_final)

        return out, cou_emb, beh_lstm_emb[0]  # BT, 8, 8


class BiLSTM(nn.Module):
    def __init__(self, x_in_dim, d_h, time_step):
        super(BiLSTM, self).__init__()
        self.d_h = d_h
        self.lstm = nn.LSTM(input_size=x_in_dim, hidden_size=d_h, bidirectional=True, batch_first=True)
        # fc
        self.fc = nn.Sequential(
            nn.Linear(d_h * 2, d_h),
            nn.BatchNorm1d(num_features=time_step, track_running_stats=False),
            nn.LeakyReLU()
        )

    def forward(self, X):
        # X: [B, T, 27]
        batch_size = X.shape[0]

        hidden_state = torch.randn(1 * 2, batch_size, self.d_h).to(X.device)
        # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.randn(1 * 2, batch_size, self.d_h).to(X.device)
        # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(X, (hidden_state, cell_state))  # [B, T, n_hidden * 2]
        H = self.fc(outputs)  # model : [B, T, d_h]
        return H


class ChannelAttention(nn.Module):
    def __init__(self, done_node_num):
        super(ChannelAttention, self).__init__()
        self.channel_dim = done_node_num

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv = nn.Sequential(
            nn.Conv2d(self.channel_dim, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, self.channel_dim, 1)
        )

        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(num_features=3, track_running_stats=False)

    def forward(self, matrix):
        """
        :param matrix: BT, 3, 9, 8  ->  BT, 9, 8  ->  BT, 1, 8  -> M * M.T  ->  BT, 8, 8
        :return: BT, 3, 1, 1
        """
        avg_out = self.conv(self.avg_pool(matrix))
        max_out = self.conv(self.max_pool(matrix))
        attention_out = self.sigmoid(self.bn(avg_out + max_out))  # BT, 3, 1, 1

        return attention_out


class SpatialAttention(nn.Module):
    def __init__(self, done_node_fea_dim):
        super(SpatialAttention, self).__init__()
        self.spatial_dim = done_node_fea_dim

        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features=1, track_running_stats=False)

    def forward(self, matrix):
        """
        :param matrix: BT, 9, 8  ->  BT, 1, 8  -> M * M.T  ->  BT, 8, 8
        :return: BT, 1, 9, 8
        """
        avg_out = torch.mean(matrix, dim=1, keepdim=True)
        max_out, _ = torch.max(matrix, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        attention_out = self.sigmoid(self.bn(self.conv(out)))  # BT, 1, 9, 8
        # matrix_out = torch.sum(torch.mul(matrix, attention_out), dim=1)
        return attention_out


class KernelM(nn.Module):
    def __init__(self, channel_dim, spatial_dim, d_h):
        super(KernelM, self).__init__()
        self.channel_fusion = ChannelAttention(channel_dim)
        self.spatial_fusion = SpatialAttention(spatial_dim)

        self.fc = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.LeakyReLU(),
            nn.Linear(d_h, d_h)
        )

    def forward(self, matrix):
        """
        :param matrix: BT, 3, 9, 8  ->  BT, 9, 8  ->  BT, 1, 8  -> M * M.T  ->  BT, 8, 8
        :return: BT, 8, 8
        """
        channel_attention = self.channel_fusion(matrix)  # BT, 3, 1, 1
        matrix_2D = torch.sum(torch.mul(matrix, channel_attention), dim=1)  # BT, 9, 8

        spatial_attention = self.spatial_fusion(matrix)  # BT, 1, 9, 8
        matrix2vector = torch.sum(torch.mul(matrix_2D, spatial_attention.squeeze(1)), dim=1, keepdim=True)  # BT, 1, 8

        vector2square = torch.bmm(matrix2vector.transpose(2, 1), matrix2vector)  # BT, 8, 8
        M = self.fc(vector2square)
        return M
