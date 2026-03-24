import torch
import torch.nn as nn
import geoopt
from geoopt.manifolds import PoincareBall, Lorentz
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, matthews_corrcoef, f1_score
import warnings
from scipy.sparse import SparseEfficiencyWarning

warnings.simplefilter('ignore', SparseEfficiencyWarning)
import manifolds
from models import encoders
from layers.layers import FermiDiracDecoder
from utils.eval_utils import MarginLoss
import torch.nn.functional as F
from utils.train_utils import get_dir_name, format_metrics
from models.models import GWTNet


# 双曲特征提取模块
class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = 'Lorentz'
        self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, x, adj):
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
            if self.manifold.name == 'Lorentz':
                x = self.manifold.expmap0(x)
        h = self.encoder.encode(x, adj)
        if self.manifold.name == 'Lorentz':
            h = self.manifold.projx(h)
        return h


class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        # self.dc = FermiDiracDecoder(r=2., t=1.)
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges
        self.loss = MarginLoss(2.)

    def decode(self, h, idx):

        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)

        # return -sqdist
        # 将距离映射到概率区间
        scale = 1.0 / h.size(1)  # 可学习的缩放参数更好
        return torch.sigmoid(-scale * sqdist)  # 范围[0,1]

    # def compute_metrics(self, embeddings, data, split):
    #
    #     if split == 'train':
    #         edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
    #     else:
    #         edges_false = data[f'{split}_edges_false']
    #     pos_scores = self.decode(embeddings, data[f'{split}_edges'])
    #     neg_scores = self.decode(embeddings, edges_false)
    #     preds = torch.stack([pos_scores, neg_scores], dim=-1)
    #
    #     loss = self.loss(preds)
    #
    #     if pos_scores.is_cuda:
    #         pos_scores = pos_scores.cpu()
    #         neg_scores = neg_scores.cpu()
    #     labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
    #     preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
    #
    #     # 计算准确率
    #     roc = roc_auc_score(labels, preds)
    #     ap = average_precision_score(labels, preds)
    #
    #
    #
    #     metrics = {'loss': loss, 'roc': roc, 'ap': ap}
    #     return metrics
    # def compute_metrics(self, embedding, edges, edges_false, n_bootstrap=200):
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

    def init_metric_dict(self):
        return {'roc': (-1, []), 'ap': (-1, []), 'accuracy': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'][0] + m1['ap'][0]) < 0.5 * (m2['roc'][0] + m2['ap'][0])


class HyperGCL(nn.Module):
    def __init__(self, c=1.0, lambda_=0.1):
        super().__init__()
        self.c = c  # 双曲空间曲率
        self.lambda_ = lambda_  # 均匀性损失权重
        self.manifold = PoincareBall(c=c)  # 使用Poincaré球模型

    def mobius_add(self, x, y):
        """实现Poincaré球模型中的Möbius加法"""
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        denominator = 1 + 2 * self.c * xy + self.c ** 2 * x2 * y2
        return ((1 + 2 * self.c * xy + self.c * y2) * x +
                (1 - self.c * x2) * y) / denominator

    # def alignment_loss(self, z1, z2):
    #     """双曲对齐损失（公式5）"""
    #     return (2 / (self.c**0.5)) * torch.mean(torch.atanh(self.c**0.5 * torch.norm(-z1 + z2, dim=1)))
    def alignment_loss(self, z1, z2):
        """修正后的双曲对齐损失（公式5）"""
        mobius_diff = self.mobius_add(-z1, z2)  # 使用Möbius加法
        norms = torch.norm(mobius_diff, dim=1)
        return (2 / (self.c ** 0.5)) * torch.mean(torch.atanh(self.c ** 0.5 * norms))

    def isotropy_loss(self, z):
        """外环各向同性损失（公式10）"""
        v = self.manifold.logmap0(z)  # 映射到切空间
        mu = torch.mean(v, dim=0)
        cov = torch.cov(v.T)
        I = torch.eye(v.size(1), device=z.device)
        kl_div = 0.5 * (torch.trace(cov) - torch.logdet(cov) - v.size(1) + torch.norm(mu) ** 2)
        return kl_div


# 全双曲神经网络模型

class FullyHyperbolicNN(nn.Module):
    def __init__(self, args, lambda_=0.1):
        super().__init__()
        self.protein_extractor = LPModel(args)
        self.manifold = self.protein_extractor.manifold  # keep k consistent with encoder manifold
        self.dropout = nn.Dropout(0.2)
        self.lambda_ = lambda_
        self.mask_prob = 0.2
        self.check_lorentz = True
    def forward(self, protein_feat, idx):
        protein_feat = self.protein_extractor.encode(protein_feat, idx)
        return protein_feat

    # def contrastive_loss(self,z1, z2, temperature=2.0):
    #     """
    #     HLCL 对比学习损失函数
    #     :param z1: 第一个视图的节点嵌入
    #     :param z2: 第二个视图的节点嵌入
    #     :param temperature: 温度参数
    #     :return: 对比损失
    #     """
    #     z1 = F.normalize(z1, p=2, dim=1)  # L2 归一化
    #     z2 = F.normalize(z2, p=2, dim=1)  # L2 归一化
    #
    #     # 计算相似度矩阵
    #     sim_matrix = torch.exp(torch.mm(z1, z2.t()) / temperature)
    #
    #     # 正样本对是同一节点的两个视图
    #     pos_sim = torch.diag(sim_matrix)
    #
    #     # 负样本对是不同节点的视图
    #     neg_sim = sim_matrix.sum(dim=1) - pos_sim
    #
    #     # InfoNCE 损失
    #     loss = -torch.log(pos_sim / neg_sim).mean()
    #     return loss
    # def contrastive_loss(self, z1, z2):
    #     # === 新增：检测 z1 / z2 是否在洛伦兹空间（双曲面） ===
    #     if self.check_lorentz:
    #         # 如果 HyperGCL 里也有曲率 c，最好对齐同一个 c
    #         # （否则你这里用 self.c 检查，HyperGCL 用别的 c 计算，会不一致）
    #         self._check_in_lorentz_space(z1, name="z1")
    #         self._check_in_lorentz_space(z2, name="z2")
    #
    #     hypergcl = HyperGCL()
    #     # 双曲对齐损失
    #     align_loss = hypergcl.alignment_loss(z1, z2)
    #     # 外环各向同性损失（对两个视图）
    #     iso_loss = hypergcl.isotropy_loss(z1) + hypergcl.isotropy_loss(z2)
    #     return align_loss + self.lambda_ * iso_loss
    import torch
    import warnings

    def contrastive_loss(self, z1, z2):
        # === 新增：检测 z1 / z2 是否在洛伦兹空间（双曲面 Hyperboloid） ===
        if getattr(self, "check_lorentz", False):
            print("check lorentz happen")
            # 你可以按需调整容差
            tol = getattr(self, "lorentz_tol", 1e-3)
            require_upper_sheet = getattr(self, "lorentz_upper_sheet", True)
            raise_on_fail = getattr(self, "lorentz_raise_on_fail", True)

            k = float(self.manifold.k)
            target = -k

            def lorentz_inner(x, y):
                # <x,y>_L = -x0*y0 + sum_{i>=1} xi*yi
                return -x[:, 0] * y[:, 0] + torch.sum(x[:, 1:] * y[:, 1:], dim=1)

            def check_lorentz(z, name):
                # shape checks
                if (not torch.is_tensor(z)) or (z.dim() != 2):
                    msg = f"[{name}] expected torch.Tensor with shape [B, d+1], got {type(z)} shape={getattr(z, 'shape', None)}"
                    if raise_on_fail: raise ValueError(msg)
                    warnings.warn(msg)
                    return

                if z.size(1) < 2:
                    msg = f"[{name}] expected second dim >= 2 for Lorentz model, got {z.size(1)}"
                    if raise_on_fail: raise ValueError(msg)
                    warnings.warn(msg)
                    return

                if not torch.isfinite(z).all():
                    msg = f"[{name}] contains NaN/Inf"
                    if raise_on_fail: raise ValueError(msg)
                    warnings.warn(msg)
                    return

                # hyperboloid constraint: <z,z>_L ≈ -1/c
                ln = lorentz_inner(z, z)  # [B]
                max_abs_err = (ln - target).abs().max().item()

                ok = max_abs_err <= tol

                # upper sheet constraint: z0 > 0
                if require_upper_sheet:
                    ok = ok and bool((z[:, 0] > 0).all().item())
                    min_z0 = z[:, 0].min().item()
                else:
                    min_z0 = None

                if not ok:
                    msg = (
                            f"[{name}] not on Lorentz hyperboloid: "
                            f"max|<z,z>_L - (-k)|={max_abs_err:.3e} (tol={tol:.3e}), k={k}"
                            + (f", min(z0)={min_z0:.3e} (require z0>0)" if require_upper_sheet else "")
                    )
                    if raise_on_fail:
                        raise ValueError(msg)
                    else:
                        warnings.warn(msg)

            check_lorentz(z1, "z1")
            check_lorentz(z2, "z2")
        hypergcl = HyperGCL()
        z1 = self.manifold.lorentz_to_poincare(z1)
        z2 = self.manifold.lorentz_to_poincare(z2)
        if getattr(self, "check_lorentz", False):
            print("check Poincaré happen")
            tol_p = getattr(self, "poincare_tol", 1e-5)
            raise_on_fail = getattr(self, "lorentz_raise_on_fail", True)
            c = float(hypergcl.c)
            max_norm = (1.0 / (c ** 0.5)) - tol_p

            def check_poincare(z, name):
                if (not torch.is_tensor(z)) or (z.dim() != 2):
                    msg = f"[{name}] expected torch.Tensor with shape [B, d], got {type(z)} shape={getattr(z, 'shape', None)}"
                    if raise_on_fail:
                        raise ValueError(msg)
                    warnings.warn(msg)
                    return

                if not torch.isfinite(z).all():
                    msg = f"[{name}] contains NaN/Inf after poincare conversion"
                    if raise_on_fail:
                        raise ValueError(msg)
                    warnings.warn(msg)
                    return

                norms = torch.norm(z, dim=1)
                if not bool((norms < max_norm).all().item()):
                    max_val = norms.max().item()
                    msg = f"[{name}] not in Poincare ball: max||z||={max_val:.3e}, limit={max_norm:.3e}, c={c}"
                    if raise_on_fail:
                        raise ValueError(msg)
                    warnings.warn(msg)

            check_poincare(z1, "z1_poincare")
            check_poincare(z2, "z2_poincare")

        # 双曲对齐损失
        align_loss = hypergcl.alignment_loss(z1, z2)

        # 外环各向同性损失（对两个视图）
        iso_loss = hypergcl.isotropy_loss(z1) + hypergcl.isotropy_loss(z2)

        return align_loss + self.lambda_ * iso_loss

    def generate_graph_features_with_dropout(self, embeddings):
        return self.dropout(embeddings)

    def safe_feature_augment(self, feature):
        #
        # # 安全的logmap0
        # feature = self.manifold.expmap0(feature)
        #
        # if torch.isnan(feature).any():
        #     print("Warning: NaN in logmap0")
        #     feature = torch.zeros_like(feature)
        #
        # # 应用掩码增强而非dropout
        # mask = (torch.rand_like(feature) > self.mask_prob).float()
        # scaling = 1.0 / (1.0 - self.mask_prob + 1e-6)
        # feature = feature * mask * scaling
        #
        # # 安全的expmap0
        # # feature = self.manifold.logmap0(feature)
        # if torch.isnan(feature).any():
        #     print("Warning: NaN in expmap0")
        #     feature = self.manifold.origin(feature.size(0), feature.size(1))

        return self.dropout(feature)

    def adj_augment(self, adj):
        adj = adj.to_dense()
        return self.dropout(adj)


#设置参数

# 数据加载
import scipy.sparse as sp
from sklearn.model_selection import KFold
from utils.data_utils import load_data_corrected, process, load_data_lp


# def train(args):
#     data = load_data(args)
#     args.n_nodes, args.feat_dim = data['features'].shape
#     args.nb_false_edges = len(data['train_edges_false'])
#     args.nb_edges = len(data['train_edges'])
#
#
#     # 初始化模型
#     model = FullyHyperbolicNN(args=args)
#     GWTmodel = GWTNet(args)
#     # 优化器
#     all_params = list(model.parameters()) + list(GWTmodel.parameters())
#     # optimizer = torch.optim.Adam(all_params, lr=1e-3)
#     optimizer = geoopt.optim.RiemannianAdam(all_params, lr=0.01)
#     # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#
#     best_val_metrics = model.protein_extractor.init_metric_dict()
#     best_test_metrics = None
#     counter = 0
#     feature_augment = model.safe_feature_augment(data['features'])
#     adj_augment = model.adj_augment(data["adj_train_norm"])    # 训练循环
#     for epoch in range(100):
#         model.train()
#         GWTmodel.train()
#         optimizer.zero_grad()
#         embeddings_origin = model(data['features'],data["adj_train_norm"])
#
#         embeddings_augment = model(feature_augment,adj_augment)
#
#
#         contrastive_loss_drug =  model.contrastive_loss(embeddings_augment,embeddings_origin)
#         #小波变换
#         embeddings = model(data['features'],data["adj_train_norm"])
#         embeddings = embeddings.detach().numpy()
#         embeddings = GWTmodel(embeddings,data["adj_train_norm"])
#
#         edges_false = data['train_edges_false'][np.random.randint(0, args.nb_false_edges, args.nb_edges)]
#         train_metrics = GWTmodel.compute_metrics(embeddings, data["train_edges"], edges_false)
#
#         #损失（小波）
#         # all_train_metrics = train_metrics['loss']
#
#         # 损失（对比）
#         # train_metrics = model.protein_extractor.compute_metrics(embeddings, data["train_edges"],data["train_edges_false"])
#         all_train_metrics = contrastive_loss_drug + train_metrics['loss']
#
#         # 损失（对比+小波）
#         # all_train_metrics = contrastive_loss_drug
#
#
#         all_train_metrics.backward()
#         optimizer.step()
#         print(f"Epoch {epoch}, Loss: {train_metrics['loss'].item()},roc:{train_metrics['roc']},ap:{train_metrics['ap']}")
#         with torch.no_grad():
#             model.eval()
#             GWTmodel.eval()
#             # embeddings = model(data['features'],data["adj_train_norm"])
#
#             # embeddings = embeddings.detach().numpy()
#             # embeddings = GWTmodel(embeddings,data["adj_train_norm"])
#
#             val_metrics = model.protein_extractor.compute_metrics(embeddings, data['val_edges'], data['val_edges_false'])
#             print(f"Epoch {epoch}, Loss: {val_metrics['loss'].item()},roc:{val_metrics['roc']},ap:{val_metrics['ap']}")
#
#             if model.protein_extractor.has_improved(best_val_metrics, val_metrics):
#                 best_test_metrics = model.protein_extractor.compute_metrics(
#                     embeddings, data['test_edges'], data['test_edges_false'])
#
#                 best_val_metrics = val_metrics
#                 counter = 0
#             else:
#                 counter += 1
#                 if counter == args.patience and epoch > args.min_epochs:
#                     print("Early stopping")
#                     break
#
#     print(" ".join(
#         ["Val set results:",
#          f"Loss: {best_val_metrics['loss'].item()},roc:{best_val_metrics['roc']},ap:{best_val_metrics['ap']}"]))
#
#     print(" ".join(
#         ["Test set results:",
#          f"Loss: {best_test_metrics['loss'].item()},roc:{best_test_metrics['roc']},ap:{best_test_metrics['ap']}"]))
import matplotlib.pyplot as plt

def cross_validation_train(args, n_folds=5):
    """交叉验证训练"""
    # 加载完整数据
    data = load_data_lp()
    original_adj = data['adj_train']
    features = data['features']

    # 获取所有边
    adj_coo = original_adj.tocoo()
    all_edges = list(zip(adj_coo.row, adj_coo.col))
    all_edges = [(u, v) for u, v in all_edges if u != v]

    # K折交叉验证
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=args.split_seed)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(all_edges)):
        print(f"\n=== Fold {fold + 1}/{n_folds} ===")

        # 分割边
        train_edges = [all_edges[i] for i in train_idx]
        test_edges = [all_edges[i] for i in test_idx]

        # 从测试边中再分割验证集
        n_val = int(len(test_edges) * 0.2)  # 20%的测试边作为验证
        val_edges = test_edges[:n_val]
        test_edges = test_edges[n_val:]

        # 创建训练邻接矩阵
        train_adj = sp.lil_matrix(original_adj.shape)
        for u, v in train_edges:
            train_adj[u, v] = 1
            train_adj[v, u] = 1
        train_adj = train_adj.tocsr()

        # 生成负样本
        def generate_false_edges(n_samples, existing_edges):
            false_edges = []
            n_nodes = original_adj.shape[0]
            existing_set = set(existing_edges)

            while len(false_edges) < n_samples:
                u, v = np.random.randint(0, n_nodes, 2)
                if u != v and (u, v) not in existing_set and (v, u) not in existing_set:
                    if original_adj[u, v] == 0:
                        false_edges.append((u, v))
                        existing_set.add((u, v))
                        existing_set.add((v, u))

            return false_edges

        train_edges_false = generate_false_edges(len(train_edges), all_edges)
        val_edges_false = generate_false_edges(len(val_edges), all_edges + train_edges_false)
        test_edges_false = generate_false_edges(len(test_edges), all_edges + train_edges_false + val_edges_false)

        # 准备fold数据
        fold_data = {
            'features': torch.tensor(features.toarray(), dtype=torch.float32),
            'adj_train': train_adj,
            'adj_train_norm': process(train_adj, features, args.normalize_adj, args.normalize_feats)[0],
            'train_edges': torch.LongTensor(train_edges),
            'train_edges_false': torch.LongTensor(train_edges_false),
            'val_edges': torch.LongTensor(val_edges),
            'val_edges_false': torch.LongTensor(val_edges_false),
            'test_edges': torch.LongTensor(test_edges),
            'test_edges_false': torch.LongTensor(test_edges_false)
        }

        args.n_nodes, args.feat_dim = fold_data['features'].shape
        args.nb_false_edges = len(fold_data['train_edges_false'])
        args.nb_edges = len(fold_data['train_edges'])

        fold_metrics = single_fold_train(args, fold_data)
        fold_results.append(fold_metrics)

        # print(f"Fold {fold + 1} Test - ROC: {fold_metrics['roc']:.4f}, AP: {fold_metrics['ap']:.4f}")
        print(fold_metrics)

    # 汇总结果
    roc_scores = [r['roc'][0] for r in fold_results]
    ap_scores = [r['ap'][0] for r in fold_results]

    def plot_cv_folds_curves(fold_results, out_path):
        from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        for i, r in enumerate(fold_results, 1):
            curve = r.get('curve_data', {})
            y_true = curve.get('labels_np')
            y_score = curve.get('preds_np')
            if y_true is None or y_score is None:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Fold {i} (AUC={roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title("(a) Receiver Operating Characteristic")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")

        plt.subplot(1, 2, 2)
        for i, r in enumerate(fold_results, 1):
            curve = r.get('curve_data', {})
            y_true = curve.get('labels_np')
            y_score = curve.get('preds_np')
            if y_true is None or y_score is None:
                continue
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            pr_auc = average_precision_score(y_true, y_score)
            plt.plot(rec, prec, label=f"Fold {i} (AP={pr_auc:.3f})")
        plt.title("(b) Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")

        plt.tight_layout()
        plt.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close()

    plot_cv_folds_curves(fold_results, "cv_test_roc_pr_folds.png")

    print(f"\n=== Cross Validation Results ===")
    print(f"ROC-AUC: {np.mean(roc_scores):.4f} ± {np.std(roc_scores):.4f}")
    print(f"Average Precision: {np.mean(ap_scores):.4f} ± {np.std(ap_scores):.4f}")

    return fold_results


def single_fold_train(args, data):
    """单折训练"""
    model = FullyHyperbolicNN(args=args)
    GWTmodel = GWTNet(args)

    all_params = list(model.parameters()) + list(GWTmodel.parameters())
    optimizer = geoopt.optim.RiemannianAdam(all_params, lr=0.01)

    best_val_metrics = model.protein_extractor.init_metric_dict()
    best_test_metrics = None
    counter = 0

    for epoch in range(5):
        model.train()
        GWTmodel.train()
        optimizer.zero_grad()

        # 前向传播
        embeddings_origin = model(data['features'], data["adj_train_norm"])

        # 对比学习
        feature_augment = model.safe_feature_augment(data['features'])
        adj_augment = model.adj_augment(data["adj_train_norm"])
        embeddings_augment = model(feature_augment, adj_augment)
        # contrastive_loss_drug = model.contrastive_loss(embeddings_augment, embeddings_origin)

        # 小波变换
        embeddings = model.manifold.logmap0(embeddings_origin).detach().numpy()
        embeddings = GWTmodel(embeddings, data["adj_train_norm"])

        # 计算损失
        edges_false = data['train_edges_false'][
            np.random.randint(0, len(data['train_edges_false']), len(data['train_edges']))]
        train_metrics = GWTmodel.compute_metrics(embeddings, data["train_edges"], edges_false)

        # contrastive_loss_drug +
        all_train_metrics =  train_metrics['loss']
        all_train_metrics.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {train_metrics['loss'].item()}, "
              f"roc:{train_metrics['roc']}, ap:{train_metrics['ap']}")

        # 验证
        if epoch % 1 == 0:
            model.eval()
            GWTmodel.eval()
            with torch.no_grad():
                embeddings = model(data['features'], data["adj_train_norm"])
                if not torch.isfinite(embeddings).all():
                    print("[Val] non-finite embeddings from model output")
                embeddings = model.manifold.logmap0(embeddings).detach().numpy()
                if not np.isfinite(embeddings).all():
                    print("[Val] non-finite embeddings after logmap0")
                embeddings = GWTmodel(embeddings, data["adj_train_norm"])
                if not torch.isfinite(embeddings).all():
                    print("[Val] non-finite embeddings after GWTmodel")

                val_metrics = GWTmodel.compute_metrics(
                    embeddings, data['val_edges'], data['val_edges_false'])
                print(f"Epoch {epoch}, Val Loss: {val_metrics['loss'].item()}, "
                      f"roc:{val_metrics['roc']}, ap:{val_metrics['ap']}")

                if model.protein_extractor.has_improved(best_val_metrics, val_metrics):
                    best_test_metrics = GWTmodel.compute_metrics(
                        embeddings, data['test_edges'], data['test_edges_false'])
                    # 🔥 新增：获取预测分数前三的蛋白质对
                    top_predictions = get_top_predictions(
                        model, GWTmodel, data, top_k=3
                    )
                    best_val_metrics = val_metrics
                    counter = 0
                else:
                    counter += 1
                    if counter == args.patience:
                        break

    def get_scores(edges, edges_false):
        embeddings = model(data['features'], data["adj_train_norm"])
        embeddings = model.manifold.logmap0(embeddings).detach().numpy()
        embeddings = GWTmodel(embeddings, data["adj_train_norm"])
        pos_scores = GWTmodel.decode(embeddings, edges)
        neg_scores = GWTmodel.decode(embeddings, edges_false)
        preds = torch.cat([pos_scores, neg_scores]).detach().numpy()
        labels = np.concatenate(
            [np.ones(pos_scores.size(0)), np.zeros(neg_scores.size(0))])
        return labels, preds

    if best_test_metrics is None:
        best_test_metrics = {}
    labels_np, preds_np = get_scores(data['test_edges'], data['test_edges_false'])
    best_test_metrics['curve_data'] = {'labels_np': labels_np, 'preds_np': preds_np}
    return best_test_metrics


def get_top_predictions(model, GWTmodel, data, top_k=3):
    """
    获取预测分数最高的蛋白质对
    """
    model.eval()
    GWTmodel.eval()

    with torch.no_grad():
        # 获取所有蛋白质的嵌入
        embeddings = model(data['features'], data["adj_train_norm"])
        embeddings = model.manifold.logmap0(embeddings).detach().numpy()
        embeddings = GWTmodel(embeddings, data["adj_train_norm"])

        # 计算所有可能的蛋白质对的预测分数
        num_proteins = embeddings.shape[0]
        all_scores = []

        # 方法1: 如果GWTmodel有直接的预测函数
        if hasattr(GWTmodel, 'predict_pair_score'):
            for i in range(num_proteins):
                for j in range(i + 1, num_proteins):  # 避免重复对
                    score = GWTmodel.predict_pair_score(embeddings, i, j)
                    all_scores.append({
                        'protein_i': i,
                        'protein_j': j,
                        'score': score,
                        'protein_i_name': data.get('protein_names', [])[
                            i] if 'protein_names' in data else f"Protein_{i}",
                        'protein_j_name': data.get('protein_names', [])[
                            j] if 'protein_names' in data else f"Protein_{j}"
                    })

        # 方法2: 使用decoder计算分数（如果可用）
        elif hasattr(GWTmodel, 'decoder'):
            # 创建所有可能的蛋白质对
            all_pairs = []
            for i in range(num_proteins):
                for j in range(i + 1, num_proteins):
                    all_pairs.append([i, j])

            if all_pairs:
                all_pairs = np.array(all_pairs)
                scores = GWTmodel.decoder(embeddings, all_pairs)

                for idx, (i, j) in enumerate(all_pairs):
                    all_scores.append({
                        'protein_i': i,
                        'protein_j': j,
                        'score': scores[idx],
                        'protein_i_name': data.get('protein_names', [])[
                            i] if 'protein_names' in data else f"Protein_{i}",
                        'protein_j_name': data.get('protein_names', [])[
                            j] if 'protein_names' in data else f"Protein_{j}"
                    })

        # 方法3: 使用相似度计算（基于嵌入向量的余弦相似度）
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(embeddings)

            for i in range(num_proteins):
                for j in range(i + 1, num_proteins):
                    score = similarity_matrix[i, j]
                    all_scores.append({
                        'protein_i': i,
                        'protein_j': j,
                        'score': score,
                        'protein_i_name': data.get('protein_names', [])[
                            i] if 'protein_names' in data else f"Protein_{i}",
                        'protein_j_name': data.get('protein_names', [])[
                            j] if 'protein_names' in data else f"Protein_{j}"
                    })

        # 按分数降序排序并返回前top_k个
        all_scores.sort(key=lambda x: x['score'], reverse=True)

        # 过滤掉训练集中已存在的相互作用（可选）
        filtered_scores = []
        train_edges_set = set(tuple(edge) for edge in data['train_edges'])

        for pred in all_scores:
            edge_tuple = (pred['protein_i'], pred['protein_j'])
            reverse_tuple = (pred['protein_j'], pred['protein_i'])

            # 如果该对不在训练集中，则保留（发现新的相互作用）
            if edge_tuple not in train_edges_set and reverse_tuple not in train_edges_set:
                filtered_scores.append(pred)

        # 如果过滤后数量不足，则使用原始排序
        top_predictions = filtered_scores[:top_k] if len(filtered_scores) >= top_k else all_scores[:top_k]

        # 打印结果
        print(f"\n🔬 Top {top_k} Predicted Novel Protein-Protein Interactions:")
        print("=" * 80)
        for rank, pred in enumerate(top_predictions, 1):
            print(f"Rank {rank}:")
            print(f"  Proteins: {pred['protein_i_name']} ↔ {pred['protein_j_name']}")
            print(f"  Prediction Score: {pred['score']:.4f}")
            print(f"  Protein IDs: {pred['protein_i']} - {pred['protein_j']}")
            print("-" * 40)

        return top_predictions


def train(args):
    """主训练函数 - 提供两种模式"""
    if args.cross_validation:
        print("Using Cross-Validation Mode")
        results = cross_validation_train(args, n_folds=5)
    else:
        print("Using Single Split Mode")
        data = load_data_corrected(args)
        args.n_nodes, args.feat_dim = data['features'].shape
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        results = single_fold_train(args, data)
        print("Test set results:")
        print(f"ROC-AUC: {results['roc']:.4f}, AP: {results['ap']:.4f}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # Add all arguments properly
    parser.add_argument('--model', type=str, default="HyboNet")
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--bias', type=int, default=1)
    parser.add_argument('--use_att', type=int, default=1)
    parser.add_argument('--local_agg', type=int, default=0)
    parser.add_argument('--val_prop', type=float, default=0.05)
    parser.add_argument('--test_prop', type=float, default=0.1)
    parser.add_argument('--use_feats', type=int, default=1)
    parser.add_argument('--normalize_feats', type=int, default=1)
    parser.add_argument('--normalize_adj', type=int, default=1)
    parser.add_argument('--split_seed', type=int, default=1234)
    parser.add_argument('--manifold', type=str, default="Lorentz")
    parser.add_argument('--act', type=str, default=None)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--task', type=str, default="lp")
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--pretrained_embeddings', type=str, default=None)
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--min_epochs', type=int, default=200)
    parser.add_argument('--scales', nargs='+', type=int, default=[1, 2, 3, 4])  # List of integers
    parser.add_argument('--data_augment ', type=str, default='')
    parser.add_argument('--drop_rate', type=float, default=0.2)
    parser.add_argument('--add_rate', type=float, default=0.1)
    parser.add_argument('--noise_level', type=float, default=0.05)

    parser.add_argument('--cross_validation', type=int, default=1, help='Use cross validation')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds')
    args = parser.parse_args()
    args.model = "HyboNet"
    args.num_layers = 2
    args.dropout = 0.0
    args.bias = 1
    args.use_att = 1
    args.local_agg = 0
    args.val_prop = 0.25
    args.test_prop = 0.25
    args.use_feats = 1
    args.normalize_feats = 1
    args.normalize_adj = 1
    args.split_seed = 523
    args.manifold = "Lorentz"
    args.act = None
    args.dim = 128
    args.task = "lp"
    args.c = 1.
    args.cuda = -1
    args.n_heads = 8
    args.alpha = 0.2
    args.pretrained_embeddings = None
    args.save = 0
    args.patience = 200
    args.min_epochs = 200
    args.scales = [1, 2, 3, 4]
    args.data_augment = ''
    args.drop_rate = 0.2
    args.add_rate = 0.1
    args.noise_level = 0.05

    args.cross_validation = 1  # 默认使用交叉验证
    args.n_folds = 5
    train(args)
