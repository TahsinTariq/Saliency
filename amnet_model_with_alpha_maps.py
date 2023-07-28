__author__ = "Jiri Fajtl"
__email__ = "ok1zjf@gmail.com"
__version__ = "6.3"
__status__ = "Research"
__date__ = "30/1/2018"
__license__ = "MIT License"

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special.basic import bi_zeros
from torchvision import models
from torch.autograd import Variable
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn


class VGG16FC(nn.Module):
    def __init__(self):
        super(VGG16FC, self).__init__()
        model = models.vgg16(pretrained=True)
        self.core_cnn = nn.Sequential(
            *list(model.features.children())[:-7]
        )  # to relu5_3`
        self.D = 512
        return

    def forward(self, x):
        x = self.core_cnn(x)
        return x


class ResNet18FC(nn.Module):
    def __init__(self):
        super(ResNet18FC, self).__init__()
        self.core_cnn = models.resnet18(pretrained=True)
        self.D = 256
        return

    def forward(self, x):
        x = self.core_cnn.conv1(x)
        x = self.core_cnn.bn1(x)
        x = self.core_cnn.relu(x)
        x = self.core_cnn.maxpool(x)

        x = self.core_cnn.layer1(x)
        x = self.core_cnn.layer2(x)
        x = self.core_cnn.layer3(x)
        return x


class ResNet50FC(nn.Module):
    def __init__(self):
        super(ResNet50FC, self).__init__()
        self.core_cnn = models.resnet50(pretrained=True)
        self.D = 1024
        return

    def forward(self, x):
        x = self.core_cnn.conv1(x)
        x = self.core_cnn.bn1(x)
        x = self.core_cnn.relu(x)
        x = self.core_cnn.maxpool(x)

        x = self.core_cnn.layer1(x)
        x = self.core_cnn.layer2(x)
        x = self.core_cnn.layer3(x)
        return x


class ResNet101FC(nn.Module):
    def __init__(self):
        super(ResNet101FC, self).__init__()
        self.core_cnn = models.resnet101(pretrained=True)
        self.D = 1024
        return

    def forward(self, x):
        x = self.core_cnn.conv1(x)
        x = self.core_cnn.bn1(x)
        x = self.core_cnn.relu(x)
        x = self.core_cnn.maxpool(x)

        x = self.core_cnn.layer1(x)
        x = self.core_cnn.layer2(x)
        x = self.core_cnn.layer3(x)
        return x


# ===============================================================================================

# Direct ResNet50 memorability estimation - no attention or RNN
class ResNet50FT(nn.Module):
    def __init__(self):
        super(ResNet50FT, self).__init__()
        self.core_cnn = models.resnet50(pretrained=True)
        self.avgpool = nn.AvgPool2d(7)
        expansion = 4
        self.fc = nn.Linear(512 * expansion, 1)
        return

    def forward(self, x):
        x = self.core_cnn.conv1(x)
        x = self.core_cnn.bn1(x)
        x = self.core_cnn.relu(x)
        x = self.core_cnn.maxpool(x)

        x = self.core_cnn.layer1(x)
        x = self.core_cnn.layer2(x)
        x = self.core_cnn.layer3(x)
        x = self.core_cnn.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        output_seq = x.unsqueeze(1)

        output = None
        alphas = None
        return output, output_seq, alphas


# ===============================================================================================
class AMemNetModel(nn.Module):
    def __init__(self, core_cnn, hps, a_res=14, a_vec_size=512):
        super(AMemNetModel, self).__init__()

        self.hps = hps
        self.use_attention = hps.use_attention
        # self.force_distribute_attention = hps.force_distribute_attention
        self.with_bn = True

        self.a_vec_size = a_vec_size  # D
        self.a_vec_num = a_res * a_res  # L

        self.seq_len = hps.seq_steps
        self.lstm_input_size = self.a_vec_size
        self.lstm_hidden_size = 1024  # H Also LSTM output
        self.lstm_layers = 1

        self.core_cnn = core_cnn

        self.inconv = nn.Conv2d(
            in_channels=core_cnn.D,
            out_channels=a_vec_size,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=True,
        )

        self.alpha_conv = nn.Conv2d(
            in_channels=3,
            out_channels=196,
            kernel_size=(224, 224),
            stride=1,
            padding=0,
            bias=True,
        )
        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(a_vec_size)

        # Layers for the h and c LSTM states
        self.hs1 = nn.Linear(
            in_features=self.a_vec_size, out_features=self.lstm_hidden_size
        )
        self.hc1 = nn.Linear(
            in_features=self.a_vec_size, out_features=self.lstm_hidden_size
        )

        # e layers
        self.e1 = nn.Linear(
            in_features=self.a_vec_size, out_features=self.a_vec_size, bias=False
        )

        # Context layers
        self.eh1 = nn.Linear(
            in_features=self.lstm_hidden_size, out_features=self.a_vec_num
        )
        self.eh3 = nn.Linear(in_features=self.a_vec_size, out_features=1, bias=False)

        # LSTM
        self.rnn = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            dropout=0.5,
            bidirectional=False,
        )

        self.rnn_2 = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            dropout=0.5,
            bidirectional=False,
        )

        # Regression Network
        self.regnet1 = nn.Linear(in_features=self.lstm_hidden_size, out_features=512)
        self.regnet4 = nn.Linear(in_features=self.regnet1.out_features, out_features=1)

        self.regnet1_alpha = nn.Linear(
            in_features=self.lstm_hidden_size, out_features=512
        )
        self.regnet4_alpha = nn.Linear(
            in_features=self.regnet1_alpha.out_features, out_features=1
        )

        self.regnet_merge = nn.Linear(in_features=2, out_features=1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.drop50 = nn.Dropout(0.5)
        self.drop80 = nn.Dropout(0.80)

        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.mask_fc = nn.Linear(1000, 1)

        # self.lgcn_1 = Lift_GCN(1024, 512)
        # self.lgcn_2 = Lift_GCN(512, 256)
        # self.lgcn_3 = Lift_GCN(256, 128)

        # self.lgcn_1 = Lift_GCN(1024, 512)
        # self.lgcn_2 = Lift_GCN(512, 256)
        # self.lgcn_3 = Lift_GCN(256, 1024)

        # self.lgcn_4 = Lift_GCN(128, 64)
        # self.lgcn_5 = Lift_GCN(64, 8)
        # self.lgcn_6 = Lift_GCN(8, 1024)

        if hps.torch_version_major == 0 and hps.torch_version_minor < 3:
            self.softmax = nn.Softmax()
        else:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x, alpha_x):
        input_images = x
        alpha_map = alpha_x

        # alpha_map = self.core_cnn(alpha_map)
        alpha_map = self.alpha_conv(alpha_map)
        alpha_map = alpha_map.view(alpha_map.size(0), 196, 1)

        # mask_features = self.model(input_images)
        # mask_features = mask_features[0]["box_features"]  # Extracting the box features
        # mask_output = self.mask_fc(mask_features)

        if not self.use_attention:
            self.alpha = torch.Tensor(x.size(0), self.a_vec_num)
            self.alpha = Variable(self.alpha)
            if self.hps.use_cuda:
                self.alpha = self.alpha.cuda()

            nn.init.constant(self.alpha, 1)
            self.alpha = self.alpha / self.a_vec_num

        # x = self.core_cnn(x)
        x = self.core_cnn(input_images)

        x = self.inconv(x)
        if self.with_bn:
            x = self.bn1(x)
        x = self.relu(x)  # -> [B, D, Ly, Lx] [B, 512, 14, 14]
        x = self.drop80(x)

        a = x.view(x.size(0), self.a_vec_size, self.a_vec_num)  # [B, D, L]

        # Extract the annotation vector
        # Mean of each feature map
        af = a.mean(2)  # [B, D]

        # Hidden states for the LSTM
        hs = self.hs1(af)  # [D->H]
        hs = self.tanh(hs)

        cs = self.hc1(af)  # [D->H]
        cs = self.tanh(cs)

        e = a.transpose(2, 1).contiguous()  # -> [B, L, D]
        e = e.view(-1, self.a_vec_size)  # a=[B, L, D] -> (-> [B*L, D])
        e = self.e1(e)  # [B*L, D] -> [B*L, D]
        e = self.relu(e)
        e = self.drop50(e)
        e = e.view(-1, self.a_vec_num, self.a_vec_size)  # -> [B, L, D]
        e = e.transpose(2, 1)  # -> [B, D, L]

        # Execute the LSTM steps
        h = hs
        rnn_state = (
            hs.expand(self.lstm_layers, hs.size(0), hs.size(1)).contiguous(),
            cs.expand(self.lstm_layers, cs.size(0), cs.size(1)).contiguous(),
        )

        rnn_state_alpha = (
            hs.expand(self.lstm_layers, hs.size(0), hs.size(1)).contiguous(),
            cs.expand(self.lstm_layers, cs.size(0), cs.size(1)).contiguous(),
        )

        steps = self.seq_len
        if steps == 0:
            steps = 1

        output_seq = [0] * steps
        alphas = [0] * steps

        output_seq_alpha = [0] * steps
        alphas_ = [0] * steps

        for i in range(steps):

            if self.use_attention:

                # Dynamic part of the alpha map from the current hidden RNN state
                if 0:
                    eh = self.eh12(h)  # -> [H -> D]
                    eh = eh.view(-1, self.a_vec_size, 1)  # [B, D, 1]
                    eh = (
                        e + eh
                    )  # [B, D, L]  + [B, D, 1]  => adds the eh vec[D] to all positions [L] of the e tensor

                if 1:
                    eh = self.eh1(h)  # -> [H -> L]
                    eh = eh.view(-1, 1, self.a_vec_num)  # [B, 1, L]
                    eh = e + eh  # [B, D, L]  + [B, 1, L]

                    # graph_eh = eh.reshape(eh.shape[0], eh.shape[2], eh.shape[1])
                    # eh = self.lgcn_1(graph_eh)
                    # eh = self.relu(eh)
                    # eh = self.drop50(eh)
                    # eh = self.lgcn_2(eh)
                    # eh = self.relu(eh)
                    # eh = self.drop50(eh)
                    # eh = self.lgcn_3(eh)
                    # eh = self.relu(eh)
                    # eh = self.drop50(eh)
                    # eh = self.lgcn_4(eh)
                    # eh = self.relu(eh)
                    # eh = self.drop50(eh)
                    # eh = self.lgcn_5(eh)
                    # eh = self.relu(eh)
                    # eh = self.drop50(eh)
                    # eh = self.lgcn_6(eh)
                    # eh = self.drop50(eh)
                    # eh = self.eh3(eh)
                    # eh = eh.reshape(eh.shape[0], eh.shape[2], eh.shape[1])

                eh = self.relu(eh)
                eh = self.drop50(eh)

                eh = eh.transpose(2, 1).contiguous()  # -> [B, L, D]
                eh = eh.view(-1, self.a_vec_size)  # -> [B*L, D]

                eh = self.eh3(eh)  # -> [B*L, 512] -> [B*L, 1]
                eh = eh.view(-1, self.a_vec_num)  # -> [B, L]

                alpha = self.softmax(eh)  # -> [B, L]

            else:
                alpha = self.alpha

            alpha_a = alpha.view(alpha.size(0), self.a_vec_num, 1)  # -> [B, L, 1]
            z = a.bmm(
                alpha_a
            )  # ->[B, D, 1] scale the location feature vectors by the alpha mask and add them (matrix mul)
            # [D, L] * [L] = [D]

            z = z.view(z.size(0), self.a_vec_size)
            z = z.expand(
                1, z.size(0), z.size(1)
            )  # Prepend a new, single dimension representing the sequence

            z_alpha = a.bmm(alpha_map)
            z_alpha = z_alpha.view(z_alpha.size(0), self.a_vec_size)
            z_alpha = z_alpha.expand(1, z_alpha.size(0), z_alpha.size(1))

            if self.seq_len == 0:
                z = z.squeeze(dim=0)
                h = self.drop50(z)

                out = self.regnet1(h)
                out = self.relu(out)
                out = self.drop50(out)
                out = self.regnet4(out)

                output_seq[0] = out
                alphas[0] = alpha.unsqueeze(1)

                z_alpha = z_alpha.squeeze(dim=0)
                h_alpha = self.drop50(z_alpha)

                out_alpha = self.regnet1(h_alpha)
                out_alpha = self.relu(out_alpha)
                out_alpha = self.drop50(out_alpha)
                out_alpha = self.regnet4(out_alpha)

                output_seq_alpha[0] = out_alpha
                alphas_[0] = alpha_map.unsqueeze(1)
                break

            # Run RNN step
            self.rnn.flatten_parameters()
            h, rnn_state = self.rnn(z, rnn_state)
            h = h.squeeze(dim=0)  # remove the seqeunce dimension
            h = self.drop50(h)

            out = self.regnet1(h)
            out = self.relu(out)
            out = self.drop50(out)
            out = self.regnet4(out)

            # Run RNN step for alpha
            self.rnn_2.flatten_parameters()
            h_alpha, rnn_state_alpha = self.rnn_2(z_alpha, rnn_state_alpha)
            h_alpha = h_alpha.squeeze(dim=0)  # remove the seqeunce dimension
            h_alpha = self.drop50(h_alpha)

            out_alpha = self.regnet1_alpha(h_alpha)
            out_alpha = self.relu(out_alpha)
            out_alpha = self.drop50(out_alpha)
            out_alpha = self.regnet4_alpha(out_alpha)

            out = torch.cat((out, out_alpha), 1)
            out = self.regnet_merge(out)

            # Store the output and the attention mask
            ind = i
            output_seq[ind] = out
            alphas[ind] = alpha.unsqueeze(1)

        output_seq = torch.cat(output_seq, 1)
        alphas = torch.cat(alphas, 1)

        output = None
        return output, output_seq, alphas

    def load_weights(self, state_dict, info=False):
        """Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. The keys of :attr:`state_dict` must
        exactly match the keys returned by this module's :func:`state_dict()`
        function.

        Arguments:
            state_dict (dict): A dict containing parameters and
                persistent buffers.
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                # raise KeyError('unexpected key "{}" in state_dict'
                #                .format(name))
                if info:
                    print(
                        'Cannot load key "{}". It does not exist in the model state_dict. Ignoring...'.format(
                            name
                        )
                    )
                # print('unexpected key "{}" in state_dict. Ignoring...'.format(name))
                continue

            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print(
                    "While copying the parameter named {}, whose dimensions in the model are"
                    " {} and whose dimensions in the checkpoint are {}, ...".format(
                        name, own_state[name].size(), param.size()
                    )
                )
                raise

        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))


# class Lift_GCN(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(Lift_GCN, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.a = nn.Parameter(torch.empty(size=(2 * self.out_features, 1))).cuda()
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)
#         self.weight_matrix_att = nn.Parameter(
#             torch.FloatTensor(self.out_features, self.out_features)
#         ).cuda()
#         self.weight = nn.Parameter(
#             torch.FloatTensor(self.in_features, self.out_features)
#         ).cuda()
#         # self.adj = adj
#         self.leaky_alpha = 0.2
#         self.leakyrelu = nn.LeakyReLU(self.leaky_alpha)
#         self.alpha = 0.7
#         self.conv1d = nn.Conv1d(
#             in_channels=self.in_features,
#             out_channels=self.out_features,
#             kernel_size=196,
#         ).cuda()
#         # self.adj = torch.ones_like(torch.empty(8, 196, 196)).cuda()
#         self.adj = nn.Parameter(torch.ones_like(torch.empty(196, 196))).cuda()
#         # self.normalized_adj = self.normalize_adjacency()
#         self.reset_parameters()

#     def normalize_adjacency(self, adj):
#         degree = torch.sum(adj, 1)
#         inv_degree = torch.div(1.0, degree)
#         inv_degree = torch.nan_to_num(inv_degree, nan=0)

#         diagonalized_degree = torch.diag_embed(inv_degree)

#         # normalized_adj = torch.bmm(diagonalized_degree, adj)
#         normalized_adj = torch.mm(diagonalized_degree, adj)
#         return normalized_adj

#     def forward(self, x):
#         # x = self.conv1d(x)
#         # self.adj = adj
#         self.normalized_adj = self.normalize_adjacency(self.adj)

#         x = torch.matmul(x, self.weight)
#         U, P = self.attention(x)

#         low_pass = torch.add(x, torch.matmul(U, x))

#         gcn = torch.matmul(self.normalized_adj, x)

#         output = (
#             self.alpha * low_pass
#             + (1 - self.alpha) * gcn
#             - (1 - self.alpha) * torch.matmul(P, low_pass)
#         )
#         # return gcn  # output
#         return output

#     def reset_parameters(self):
#         torch.nn.init.xavier_uniform_(self.weight_matrix_att)
#         torch.nn.init.xavier_uniform_(self.weight)

#     def attention(self, feature):
#         # torch.autograd.set_detect_anomaly(True)
#         # feature = 0.9*feature + 0.1*h0
#         feature = torch.matmul(feature, self.weight_matrix_att)
#         feat_1 = torch.matmul(feature, self.a[: self.out_features, :].clone())
#         feat_2 = torch.matmul(feature, self.a[self.out_features :, :].clone())

#         # broadcast add
#         # e = feat_1 + feat_2.T
#         e = feat_1 + torch.transpose(feat_2, 1, 2)

#         e = self.leakyrelu(e)

#         zero_vec = -9e15 * torch.ones_like(e)
#         # att = torch.where(self.adj > 0, e, zero_vec)
#         att = e
#         att = F.softmax(att, dim=2).clone()

#         U = att.clone()
#         P = 0.5 * U.clone()
#         # att_arr = att.cpu().detach().numpy()
#         # adj_nonzero_ind = self.raf_adj.nonzero()
#         # adj_nonzero_ind = att.nonzero(as_tuple=True)
#         return U, P
