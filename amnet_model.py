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

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


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

        # ----------------------------------------
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model

        cfg.MODEL.WEIGHTS = "./model_final_f10217.pkl"

        self.maskrcnn_model = build_model(cfg)
        DetectionCheckpointer(self.maskrcnn_model).load(cfg.MODEL.WEIGHTS)
        # self.maskrcnn_model.train(False)
        # self.maskrcnn_model.eval()
        self.features_0 = {}

        self.maskrcnn_model.roi_heads.box_head.fc_relu2.register_forward_hook(
            self.get_features_0("feats")
        )

        # ----------------------------------------
        # GEMM IMPLEMENTATION
        # ----------------------------------------
        self.softmax = nn.Softmax(
            dim=2
        )  # Since current shape is [N, 1, 1000, 1], softmax along dim = 2

        self.gemm_conv1 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=(1, 1024)
        )
        self.gemm_conv2 = nn.Conv2d(1, 1, (1, 1024))
        # self.gemm_conv3 = nn.Conv2d(4, 1)

        self.gemm_l1 = nn.Linear(1024, 512)
        self.gemm_l2 = nn.Linear(512, 4)
        self.gemm_l3 = nn.Linear(4, 1)

        self.gemm_l4 = nn.Linear(1024, 512)
        self.gemm_l5 = nn.Linear(512, 4)
        self.gemm_l6 = nn.Linear(4, 1)

        self.gemm_relu_1 = nn.ReLU()
        self.gemm_relu_2 = nn.ReLU()
        self.gemm_relu_3 = nn.ReLU()
        self.gemm_relu_4 = nn.ReLU()

        self.gemm_drop80_1 = nn.Dropout(p=0.8)
        self.gemm_drop80_2 = nn.Dropout(p=0.8)
        self.gemm_drop80_3 = nn.Dropout(p=0.8)
        self.gemm_drop80_4 = nn.Dropout(p=0.8)
        self.gemm_drop80_5 = nn.Dropout(p=0.8)
        # ----------------------------------------

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

    def get_features_0(self, name):
        def hook(model, input, output):
            # features[name] = output.detach()
            self.features_0[name] = output

        return hook

    def forward(self, x, alpha_x):
        input_images = x
        batch_size = x.shape[0]
        alpha_map = alpha_x
        # =====================================================================
        self.maskrcnn_model.eval()
        features = []
        for img_tensor in input_images:
            inputs = [{"image": img_tensor}]
            outputs = self.maskrcnn_model(inputs)
            s0 = self.features_0["feats"].shape[0]
            if s0 < 1000:
                features.append(
                    F.pad(
                        self.features_0["feats"], (0, 0, 0, 1000 - s0), "constant", 0
                    )  # reflection padding is not implemented. YET!
                )
            else:
                features.append(self.features_0["feats"])

        # inputs = [{"image": img_tensor} for img_tensor in input_images]
        features = torch.cat((features), 0)
        features = features.view(
            batch_size, 1, 1000, 1024
        )  # shape = [N, 1, 1000, 1024]
        # =====================================================================
        gemm_f = self.gemm_conv1(features)  # shape = [N, 1, 1000, 1]
        gemm_f = self.softmax(gemm_f)
        gemm_f = gemm_f.reshape((batch_size, 1000))  # shape = [N, 1000]
        f2 = features.reshape((batch_size, 1000, 1024, 1))
        z_1 = torch.einsum("nhwc,nh->nwc", f2, gemm_f)

        z_1 = z_1.view((batch_size, 1024))
        h_1 = self.gemm_l1(z_1)
        h_1 = self.gemm_relu_1(h_1)
        h_1 = self.gemm_drop80_1(h_1)

        h_1 = self.gemm_l2(h_1)
        h_1 = self.gemm_relu_2(h_1)
        h_1 = self.gemm_drop80_2(h_1)

        pred_1 = self.gemm_l3(h_1)

        alpha_1 = torch.einsum("nhwc,nh->nhwc", f2, gemm_f)
        alpha_1 = alpha_1.reshape((batch_size, 1, 1000, 1024))

        att_2 = self.gemm_conv2(alpha_1)
        att_2 = self.softmax(att_2)
        att_2 = att_2.reshape((batch_size, 1000))
        alpha_1 = alpha_1.reshape((batch_size, 1000, 1024, 1))
        z_2 = torch.einsum("nhwc,nh->nwc", alpha_1, att_2)
        z_2 = z_2.view((batch_size, 1024))

        h_2 = self.gemm_l4(z_2)
        h_2 = self.gemm_relu_3(h_2)
        h_2 = self.gemm_drop80_3(h_2)

        h_2 = self.gemm_l5(h_2)
        h_2 = self.gemm_relu_4(h_2)
        h_2 = self.gemm_drop80_4(h_2)

        pred_2 = self.gemm_l6(h_1)

        pred = torch.mean(torch.cat((pred_1, pred_2), 1), 1, True)
        # =====================================================================

        output_seq = pred
        alphas = alpha_map

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
