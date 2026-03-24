import torch
import numpy as np
import torch.nn as nn
from scipy.sparse import  diags
from scipy.sparse.linalg import inv
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score,matthews_corrcoef,f1_score
from scipy.sparse import csc_matrix
# import numpy as np
# from scipy.sparse import diags
# from scipy.linalg import inv
# class GraphWaveletTransform(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#
#     def random_walk_matrix(self, A):
#         """
#         构建随机游走矩阵P
#         :param x:邻接矩阵
#         :return:随机游走矩阵P
#         """
#         n = A.shape[0]
#         A_dense = A.to_dense()
#         A_self_loops = A_dense + np.eye(n)
#         d = diags(np.array(A_self_loops.sum(axis=1)).flatten())# 度矩阵
#         P = inv(d) @ A_self_loops  # 随机游走矩阵 P = D^{-1} A
#         return P
#
#
#
#     def graph_wavelet_transform(self,P, scales):
#         """
#         图小波变换（基于随机游走扩散）
#         :param P:随机游走矩阵
#         :param scales: 尺度列表
#         :return:小波系数列表
#         """
#         wavelets = []
#         for scale in scales:
#             T = np.linalg.matrix_power(P, scale)  # 扩散算子 T = P^scale
#             wavelets.append(T)
#         return wavelets
#
#
#
#     def extract_features(self,wavelets, X):
#         """
#         提取图小波特征
#         :param wavelets: 小波系数列表
#         :param X: 节点特征矩阵
#         :return: 图小波特征
#         """
#         features = []
#         for wavelet in wavelets:
#             features.append(wavelet @ X)  # 将小波系数应用于节点特征
#         return np.hstack(features)  # 拼接所有尺度的特征


# class GWTNet(nn.Module):
#     def __init__(self,args):
#         self.scales = args.scales
#         super().__init__()
#         self.graphwavelettransform = GraphWaveletTransform()
#         self.endfeature = nn.Linear(381,128)
#         self.normalize = nn.LayerNorm(len(args.scales)*128)
#
#     def forward(self,featrue,X):
#
#         P = self.graphwavelettransform.random_walk_matrix(X)
#         wavelets = self.graphwavelettransform.graph_wavelet_transform(P,self.scales)
#         GWTfeature = self.graphwavelettransform.extract_features(wavelets,featrue)
#
#
#         # return self.normalize(self.endfeature(torch.tensor(GWTfeature, dtype=torch.float32)))
#
#         return self.normalize(torch.tensor(GWTfeature, dtype=torch.float32))
#     def decode(self, h, idx):
#         h = self.normalize(h)
#         emb_in = h[idx[:, 0], :]
#         emb_out = h[idx[:, 1], :]
#         dist = (emb_in * emb_out).sum(dim=1)
#         return dist
#
#
#     # def compute_metrics(self,embedding,edges,edges_false):
#     #     pos_scores = self.decode(embedding, edges)
#     #     neg_scores = self.decode(embedding, edges_false)
#     #     preds = torch.cat([pos_scores, neg_scores])
#     #     pos_labels = torch.ones(pos_scores.size(0), dtype=torch.float)
#     #     neg_labels = torch.zeros(neg_scores.size(0), dtype=torch.float)
#     #     labels = torch.cat([pos_labels, neg_labels])
#     #     loss = F.binary_cross_entropy_with_logits(preds, labels)
#     #     preds_list = preds.tolist()
#     #     labels_list = labels.tolist()
#     #
#     #
#     #     roc = roc_auc_score(labels_list, preds_list)
#     #     ap = average_precision_score(labels_list, preds_list)
#     #     metrics = {'loss': loss, 'roc': roc, 'ap': ap}
#     #     return metrics
#     def compute_metrics(self, embedding, edges, edges_false, n_bootstrap=100):
#         """计算指标并执行Bootstrap显著性检验"""
#         # 原始预测
#         pos_scores = self.decode(embedding, edges)
#         neg_scores = self.decode(embedding, edges_false)
#         preds = torch.cat([pos_scores, neg_scores])
#         labels = torch.cat([torch.ones(pos_scores.size(0)),
#                             torch.zeros(neg_scores.size(0))])
#
#         # 转换为numpy
#         preds_np = preds.detach().numpy()
#         labels_np = labels.numpy()
#
#         # 原始指标
#         roc = roc_auc_score(labels_np, preds_np)
#         ap = average_precision_score(labels_np, preds_np)
#
#         # Bootstrap采样
#         def bootstrap_metric(metric_func):
#             scores = []
#             n_samples = len(labels_np)
#             for _ in range(n_bootstrap):
#                 indices = np.random.choice(n_samples, n_samples, replace=True)
#                 scores.append(metric_func(labels_np[indices], preds_np[indices]))
#             ci = np.percentile(scores, [2.5, 97.5])
#             p_value = np.mean(np.array(scores) <= 0.5)  # 针对AUC/AP的零假设
#             return ci, p_value
#
#         roc_ci, roc_p = bootstrap_metric(roc_auc_score)
#         ap_ci, ap_p = bootstrap_metric(average_precision_score)
#
#         return {
#             'loss': F.binary_cross_entropy_with_logits(preds, labels),
#             'roc': (roc, roc_ci, roc_p),
#             'ap': (ap, ap_ci, ap_p)
#         }




class GatedWaveletTransform(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.scales = args.scales
        self.num_scales = len(args.scales)
        self.feature_dim = args.dim

        # 门控机制参数
        self.scale_weights = nn.Parameter(torch.randn(1, self.num_scales))
        self.gru = nn.GRUCell(self.feature_dim, self.feature_dim)

        # 归一化层
        self.layer_norm = nn.LayerNorm(self.feature_dim)

        # 权重初始化
        nn.init.xavier_uniform_(self.scale_weights)

    def random_walk_matrix(self, A):
        """
        构建随机游走矩阵P
        :param x:邻接矩阵
        :return:随机游走矩阵P
        """
        n = A.shape[0]
        A_dense = A.to_dense()
        A_self_loops = A_dense + np.eye(n)
        d = diags(np.array(A_self_loops.sum(axis=1)).flatten())  # 度矩阵
        # P = inv(d) @ A_self_loops  # 随机游走矩阵 P = D^{-1} A
        degrees = A_self_loops.sum(dim=1)
        D_inv_sqrt = torch.diag(degrees.pow(-0.5))
        P = D_inv_sqrt @ A_self_loops @ D_inv_sqrt
        return P

    def compute_wavelets(self, P, scales):
        wavelets = []
        for scale in scales:
            T = np.linalg.matrix_power(P, scale)
            T_tensor = torch.from_numpy(T).float()  # 使用.float()确保是浮点类型
            wavelets.append(T_tensor)
            # wavelets.append(T)
        return wavelets

    def gated_fusion(self, wavelet_features,adj):
        """
        门控多尺度特征融合
        Z = ∑ softmax(θ_s) ⋅ ϕ(H(s))
        """
        # 计算各尺度权重 - 现在weights是(1, num_scales)形状
        scale_weights = F.softmax(self.scale_weights, dim=1).squeeze(0)  # 压缩成1维

        # 初始化融合特征
        fused_feature = torch.zeros_like(wavelet_features[0])
        # 加权融合
        for i, (weight, feature) in enumerate(zip(scale_weights, wavelet_features)):

            # 层归一化
            norm_feature = self.layer_norm(feature)

            # GRU门控
            if i == 0:
                gru_out = self.gru(norm_feature, torch.zeros_like(norm_feature))
            else:
                gru_out = self.gru(norm_feature, gru_out)

            # 加权融合
            fused_feature += weight * gru_out
        return fused_feature

    def forward(self, features, adj):
        # 转换为PyTorch张量
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        if not isinstance(adj, torch.Tensor):
            adj = torch.tensor(adj.toarray(), dtype=torch.float32)

        # 计算随机游走矩阵
        P = self.random_walk_matrix(adj)

        # 计算多尺度小波
        wavelets = self.compute_wavelets(P, self.scales)

        # 提取小波特征

        wavelet_features = [w @ features for w in wavelets]
        # 门控多尺度融合
        fused_features = self.gated_fusion(wavelet_features,adj)

        return fused_features


class GWTNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gated_wavelet = GatedWaveletTransform(args)
        self.normalize = nn.LayerNorm(128)

    def forward(self, features, adj):
        features = self.gated_wavelet(features, adj)
        return self.normalize(features)

    def decode(self, h, idx):
        h = self.normalize(h)
        h = F.normalize(h, p=2, dim=1, eps=1e-8)
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        dist = (emb_in * emb_out).sum(dim=1)
        return dist


    # def compute_metrics(self, embedding, edges, edges_false, n_bootstrap=100):
    #     """计算指标并执行Bootstrap显著性检验"""
    #     # 原始预测
    #     pos_scores = self.decode(embedding, edges)
    #     neg_scores = self.decode(embedding, edges_false)
    #     preds = torch.cat([pos_scores, neg_scores])
    #     labels = torch.cat([torch.ones(pos_scores.size(0)),
    #                         torch.zeros(neg_scores.size(0))])
    #
    #     # 转换为numpy
    #     preds_np = preds.detach().numpy()
    #     labels_np = labels.numpy()
    #
    #     # 原始指标
    #     roc = roc_auc_score(labels_np, preds_np)
    #     ap = average_precision_score(labels_np, preds_np)
    #
    #     # Bootstrap采样
    #     def bootstrap_metric(metric_func):
    #         scores = []
    #         n_samples = len(labels_np)
    #         for _ in range(n_bootstrap):
    #             indices = np.random.choice(n_samples, n_samples, replace=True)
    #             scores.append(metric_func(labels_np[indices], preds_np[indices]))
    #         ci = np.percentile(scores, [2.5, 97.5])
    #         p_value = np.mean(np.array(scores) <= 0.5)  # 针对AUC/AP的零假设
    #         return ci, p_value
    #
    #     roc_ci, roc_p = bootstrap_metric(roc_auc_score)
    #     ap_ci, ap_p = bootstrap_metric(average_precision_score)
    #
    #     return {
    #         'loss': F.binary_cross_entropy_with_logits(preds, labels),
    #         'roc': (roc, roc_ci, roc_p),
    #         'ap': (ap, ap_ci, ap_p)
    #     }
    def compute_metrics(self, embedding, edges, edges_false, n_bootstrap=200):

        """计算指标并执行Bootstrap显著性检验"""
        # 原始预测
        pos_scores = self.decode(embedding, edges)
        neg_scores = self.decode(embedding, edges_false)
        preds = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones(pos_scores.size(0)),
                            torch.zeros(neg_scores.size(0))])

        if not torch.isfinite(preds).all():
            min_val = preds[torch.isfinite(preds)].min().item() if torch.isfinite(preds).any() else float("nan")
            max_val = preds[torch.isfinite(preds)].max().item() if torch.isfinite(preds).any() else float("nan")
            print(f"[GWTNet] non-finite preds detected; min={min_val:.3e}, max={max_val:.3e}")
            preds = torch.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=-1.0)

        # 转换为numpy
        preds_np = preds.detach().numpy()
        labels_np = labels.numpy()

        # 计算预测类别（用于F1和MCC）
        preds_binary = (preds_np > 0.5).astype(int)

        # 原始指标
        roc = roc_auc_score(labels_np, preds_np)
        ap = average_precision_score(labels_np, preds_np)
        f1 = f1_score(labels_np, preds_binary)
        mcc = matthews_corrcoef(labels_np, preds_binary)

        # Bootstrap采样函数
        def bootstrap_metric(metric_func, use_prob=True):
            scores = []
            n_samples = len(labels_np)
            for _ in range(n_bootstrap):
                indices = np.random.choice(n_samples, n_samples, replace=True)
                if use_prob:
                    # 用于ROC AUC, AP等需要概率分数的指标
                    scores.append(metric_func(labels_np[indices], preds_np[indices]))
                else:
                    # 用于F1, MCC等需要二分类预测的指标
                    preds_binary_bootstrap = (preds_np[indices] > 0.5).astype(int)
                    scores.append(metric_func(labels_np[indices], preds_binary_bootstrap))
            ci = np.percentile(scores, [2.5, 97.5])
            # 针对不同指标的零假设
            if metric_func.__name__ in ['roc_auc_score', 'average_precision_score']:
                p_value = np.mean(np.array(scores) <= 0.5)  # AUC/AP的零假设
            elif metric_func.__name__ == 'f1_score':
                p_value = np.mean(np.array(scores) <= 0.0)  # F1的零假设
            else:  # MCC
                p_value = np.mean(np.array(scores) <= 0.0)  # MCC的零假设
            return ci, p_value

        # 对所有指标进行Bootstrap
        roc_ci, roc_p = bootstrap_metric(roc_auc_score, use_prob=True)
        ap_ci, ap_p = bootstrap_metric(average_precision_score, use_prob=True)
        f1_ci, f1_p = bootstrap_metric(f1_score, use_prob=False)
        mcc_ci, mcc_p = bootstrap_metric(matthews_corrcoef, use_prob=False)

        return {
            'loss': F.binary_cross_entropy_with_logits(preds, labels),  # 注意：这里用binary_cross_entropy而不是with_logits
            'roc': (roc, roc_ci, roc_p),
            'ap': (ap, ap_ci, ap_p),
            'f1': (f1, f1_ci, f1_p),
            'mcc': (mcc, mcc_ci, mcc_p)
        }


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from scipy.sparse import diags
# from geoopt import Lorentz
# from geoopt.manifolds.lorentz import math as lmath
#
#
# class GatedWaveletTransform(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.scales = args.scales
#         self.num_scales = len(args.scales)
#         self.feature_dim = args.dim
#         self.manifold = Lorentz(k=1.0)  # Lorentz模型，曲率k=1.0
#
#         # 门控机制参数（在切空间处理）
#         self.scale_weights = nn.Parameter(torch.randn(1, self.num_scales))
#         self.gru = nn.GRUCell(self.feature_dim, self.feature_dim)
#         self.layer_norm = nn.LayerNorm(self.feature_dim)
#         nn.init.xavier_uniform_(self.scale_weights)
#
#     def random_walk_matrix(self, A):
#         """构建随机游走矩阵P（欧式空间计算）"""
#         n = A.shape[0]
#         A_dense = A.to_dense() if A.is_sparse else A
#         A_self_loops = A_dense + torch.eye(n, device=A.device)
#         degrees = A_self_loops.sum(dim=1)
#         D_inv_sqrt = torch.diag(degrees.pow(-0.5))
#         P = D_inv_sqrt @ A_self_loops @ D_inv_sqrt
#         return P
#
#     def lorentz_matrix_multiply(self, P, x):
#         """Lorentz空间矩阵乘法"""
#         # 将欧式矩阵P投影到切空间
#         P_tangent = self.manifold.logmap0(P)
#         # 在切空间进行矩阵乘法
#         prod_tangent = P_tangent @ self.manifold.logmap0(x)
#         # 映射回Lorentz空间
#         return self.manifold.expmap0(prod_tangent)
#
#     def compute_wavelets(self, P, scales):
#         """计算多尺度小波（在切空间）"""
#         wavelets = []
#         P_tangent = self.manifold.logmap0(P)
#         for scale in scales:
#             # 在切空间进行矩阵幂运算
#             T_tangent = torch.matrix_power(P_tangent, scale)
#             wavelets.append(T_tangent)
#         return wavelets
#
#     def gated_fusion(self, wavelet_features):
#         """门控多尺度特征融合（在切空间处理）"""
#         scale_weights = F.softmax(self.scale_weights, dim=1).squeeze(0)
#         fused_feature = torch.zeros_like(wavelet_features[0])
#
#         for i, (weight, feature) in enumerate(zip(scale_weights, wavelet_features)):
#             # 将Lorentz特征投影到切空间处理
#             feature_tangent = self.manifold.logmap0(feature)
#             norm_feature = self.layer_norm(feature_tangent)
#
#             if i == 0:
#                 gru_out = self.gru(norm_feature, torch.zeros_like(norm_feature))
#             else:
#                 gru_out = self.gru(norm_feature, gru_out)
#
#             fused_feature += weight * gru_out
#
#         # 将融合结果映射回Lorentz空间
#         return self.manifold.expmap0(fused_feature)
#
#     def forward(self, features, adj):
#         """处理Lorentz空间特征"""
#         # 计算随机游走矩阵（欧式空间）
#         P = self.random_walk_matrix(adj)
#
#         # 计算多尺度小波
#         wavelets = self.compute_wavelets(P, self.scales)
#         # 提取小波特征
#         wavelet_features = []
#         for w in wavelets:
#             # 在切空间进行矩阵乘法
#             feat_tangent = self.manifold.logmap0(features)
#             prod_tangent = w @ feat_tangent
#             # 映射回Lorentz空间
#             prod_lorentz = self.manifold.expmap0(prod_tangent)
#             wavelet_features.append(prod_lorentz)
#
#         # 门控多尺度融合
#         return self.gated_fusion(wavelet_features)
#
#
# class GWTNet(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.gated_wavelet = GatedWaveletTransform(args)
#         self.manifold = Lorentz(k=1.0)
#
#     def forward(self, features, adj):
#         features = self.gated_wavelet(features, adj)
#         return features  # 返回Lorentz空间特征
#
#     def decode(self, h, idx):
#         """使用Lorentz距离作为解码"""
#         emb_in = h[idx[:, 0], :]
#         emb_out = h[idx[:, 1], :]
#         dist = -self.manifold.dist(emb_in, emb_out)  # 距离越小，分数越高
#         return dist
#
#     def compute_metrics(self, embedding, edges, edges_false, n_bootstrap=100):
#         """计算指标（使用Lorentz距离）"""
#         pos_scores = self.decode(embedding, edges)
#         neg_scores = self.decode(embedding, edges_false)
#         preds = torch.cat([pos_scores, neg_scores])
#         labels = torch.cat([torch.ones(pos_scores.size(0)),
#                             torch.zeros(neg_scores.size(0))])
#
#         preds_np = preds.detach().cpu().numpy()
#         labels_np = labels.cpu().numpy()
#
#         roc = roc_auc_score(labels_np, preds_np)
#         ap = average_precision_score(labels_np, preds_np)
#
#         def bootstrap_metric(metric_func):
#             scores = []
#             n_samples = len(labels_np)
#             for _ in range(n_bootstrap):
#                 indices = np.random.choice(n_samples, n_samples, replace=True)
#                 scores.append(metric_func(labels_np[indices], preds_np[indices]))
#             ci = np.percentile(scores, [2.5, 97.5])
#             p_value = np.mean(np.array(scores) <= 0.5)
#             return ci, p_value
#
#         roc_ci, roc_p = bootstrap_metric(roc_auc_score)
#         ap_ci, ap_p = bootstrap_metric(average_precision_score)
#
#         return {
#             'loss': F.binary_cross_entropy_with_logits(preds, labels),
#             'roc': (roc, roc_ci, roc_p),
#             'ap': (ap, ap_ci, ap_p)
#         }
