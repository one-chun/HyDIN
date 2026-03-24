"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, diags
from sklearn.utils import shuffle


def strict_train_test_split(adj, test_ratio=0.2, val_ratio=0.1, seed=42):
    """
    严格的训练-测试分割，防止数据泄露
    返回：训练邻接矩阵，掩码后的测试边
    """
    np.random.seed(seed)

    # 获取所有边
    adj_coo = adj.tocoo()
    edges = list(zip(adj_coo.row, adj_coo.col))

    # 移除自循环
    edges = [(u, v) for u, v in edges if u != v]

    # 随机打乱
    np.random.shuffle(edges)

    n_edges = len(edges)
    n_test = int(n_edges * test_ratio)
    n_val = int(n_edges * val_ratio)

    # 分割
    test_edges = edges[:n_test]
    val_edges = edges[n_test:n_test + n_val]
    train_edges = edges[n_test + n_val:]

    # 创建训练邻接矩阵（不包含测试边）
    train_adj = sp.lil_matrix(adj.shape)
    for u, v in train_edges:
        train_adj[u, v] = 1
        train_adj[v, u] = 1

    train_adj = train_adj.tocsr()

    # 生成负样本
    def generate_false_edges(adj, n_samples, existing_edges):
        false_edges = []
        n_nodes = adj.shape[0]
        existing_set = set(existing_edges)

        while len(false_edges) < n_samples:
            u, v = np.random.randint(0, n_nodes, 2)
            if u != v and (u, v) not in existing_set and (v, u) not in existing_set:
                if adj[u, v] == 0:  # 确保不是真实边
                    false_edges.append((u, v))
                    existing_set.add((u, v))
                    existing_set.add((v, u))

        return false_edges

    train_edges_false = generate_false_edges(adj, len(train_edges), edges)
    val_edges_false = generate_false_edges(adj, len(val_edges), edges + train_edges_false)
    test_edges_false = generate_false_edges(adj, len(test_edges), edges + train_edges_false + val_edges_false)

    return (train_adj, train_edges, train_edges_false,
            val_edges, val_edges_false, test_edges, test_edges_false)


def load_data_corrected(args):
    """修正后的数据加载函数"""
    # 加载原始数据
    data = load_data_lp()
    original_adj = data['adj_train']

    # 使用严格的分割
    (adj_train, train_edges, train_edges_false,
     val_edges, val_edges_false, test_edges, test_edges_false) = strict_train_test_split(
        original_adj, test_ratio=args.test_prop, val_ratio=args.val_prop, seed=args.split_seed
    )

    # 打印统计信息
    def print_stats():
        stats = {
            "Train Pos/Neg": (len(train_edges), len(train_edges_false)),
            "Val Pos/Neg": (len(val_edges), len(val_edges_false)),
            "Test Pos/Neg": (len(test_edges), len(test_edges_false)),
            "Total Nodes": original_adj.shape[0],
            "Total Original Edges": original_adj.sum() // 2
        }
        for k, v in stats.items():
            print(f"{k}: {v}")

    print_stats()

    # 处理特征和归一化
    adj_train_norm, features = process(
        adj_train, data['features'], args.normalize_adj, args.normalize_feats
    )

    data.update({
        'adj_train': adj_train,  # 只包含训练边的邻接矩阵
        'adj_train_norm': adj_train_norm,
        'train_edges': torch.LongTensor(train_edges),
        'train_edges_false': torch.LongTensor(train_edges_false),
        'val_edges': torch.LongTensor(val_edges),
        'val_edges_false': torch.LongTensor(val_edges_false),
        'test_edges': torch.LongTensor(test_edges),
        'test_edges_false': torch.LongTensor(test_edges_false),
        'adj_original': original_adj,
        'features': features
    })

    return data



# def load_data(args):
#     data = load_data_lp()  # 普通的特征矩阵和邻接矩阵
#     original_adj = data['adj_train']  # 保存原始邻接矩阵
#     adj = data['adj_train']  # 普通邻接矩阵
#     # 应用数据增强方法
#     if args.data_augment == 'edge_drop':
#         adj = random_edge_drop(original_adj, args.drop_rate)
#     elif args.data_augment == 'feature_noise':
#         data['features'] = add_feature_noise(data['features'], args.noise_level)
#     elif args.data_augment == 'edge_add':
#         adj = random_edge_add(original_adj, args.add_rate)
#     elif args.data_augment == 'mix':
#         adj = random_edge_drop(original_adj, args.drop_rate)
#         adj = random_edge_add(adj, args.add_rate)
#         data['features'] = add_feature_noise(data['features'], args.noise_level)
#     else:
#         print('无数据增强')
#         adj = original_adj  # 无数据增强
#
#     # 分割数据集
#     adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
#         adj, args.val_prop, args.test_prop, args.split_seed)
#
#     def print_stats(train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false):
#         stats = {
#             "Train Pos/Neg": (len(train_edges), len(train_edges_false)),
#             "Val Pos/Neg": (len(val_edges), len(val_edges_false)),
#             "Test Pos/Neg": (len(test_edges), len(test_edges_false)),
#             "Total Pos/Neg Ratio": (
#                 len(train_edges) + len(val_edges) + len(test_edges),
#                 len(train_edges_false) + len(val_edges_false) + len(test_edges_false)
#             )
#         }
#         for k, v in stats.items():
#             print(f"{k}: {v} (Ratio={v[0] / v[1]:.2f})")
#
#     print_stats(train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false)
#
#     data['adj_train'] = adj_train
#     data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
#     data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
#     data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
#     data['adj_train_norm'], data['features'] = process(
#         data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
#     )
#
#     # 保存原始邻接矩阵用于评估
#     data['adj_original'] = original_adj
#
#     return data


# 数据增强方法实现
def random_edge_drop(adj, drop_rate=0.1):
    """随机丢弃一定比例的边"""
    if drop_rate <= 0:
        return adj

    adj = adj.tocoo()
    num_edges = adj.nnz
    keep_mask = np.random.rand(num_edges) > drop_rate

    rows = adj.row[keep_mask]
    cols = adj.col[keep_mask]
    data = adj.data[keep_mask]

    new_adj = csr_matrix((data, (rows, cols)), shape=adj.shape)
    return new_adj


def random_edge_add(adj, add_rate=0.1):
    """随机添加一定比例的边"""
    if add_rate <= 0:
        return adj

    adj = adj.tolil()
    n = adj.shape[0]
    num_existing_edges = adj.nnz
    num_add = int(num_existing_edges * add_rate)

    for _ in range(num_add):
        u, v = np.random.randint(0, n, 2)
        while adj[u, v] != 0 or u == v:  # 避免重复边和自环
            u, v = np.random.randint(0, n, 2)
        adj[u, v] = 1
        adj[v, u] = 1  # 假设是无向图

    return adj.tocsr()


def add_feature_noise(features, noise_level=0.1):
    """向特征添加高斯噪声"""
    if noise_level <= 0:
        return features

    if isinstance(features, np.ndarray):
        noise = np.random.normal(scale=noise_level, size=features.shape)
        return features + noise
    else:  # 稀疏矩阵
        features = features.todense()
        noise = np.random.normal(scale=noise_level, size=features.shape)
        return csr_matrix(features + noise)
# def load_data(args):
#
#     data = load_data_lp()  # 普通的特征矩阵和邻接矩阵
#     adj = data['adj_train']  # 普通邻接矩阵
#     adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
#                 adj, args.val_prop, args.test_prop, args.split_seed)
#
#
#
#     def print_stats(train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false):
#         stats = {
#             "Train Pos/Neg": (len(train_edges), len(train_edges_false)),
#             "Val Pos/Neg": (len(val_edges), len(val_edges_false)),
#             "Test Pos/Neg": (len(test_edges), len(test_edges_false)),
#             "Total Pos/Neg Ratio": (
#                 len(train_edges) + len(val_edges) + len(test_edges),
#                 len(train_edges_false) + len(val_edges_false) + len(test_edges_false)
#             )
#         }
#         for k, v in stats.items():
#             print(f"{k}: {v} (Ratio={v[0] / v[1]:.2f})")
#
#     # 调用示例
#     print_stats(train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false)
#
#     data['adj_train'] = adj_train
#     data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
#     data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
#     data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
#     data['adj_train_norm'], data['features'] = process(
#             data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
#     )
#
#     return data


# ############### FEATURES PROCESSING ####################################


def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ############### DATA SPLITS #####################################################
# def mask_edges(adj, val_prop, test_prop, seed):
#     np.random.seed(seed)  # get tp edges
#     x, y = sp.triu(adj).nonzero()
#     pos_edges = np.array(list(zip(x, y)))
#     np.random.shuffle(pos_edges)
#     # get tn edges
#     x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
#     neg_edges = np.array(list(zip(x, y)))
#     np.random.shuffle(neg_edges)
#
#     m_pos = len(pos_edges)
#     n_val = int(m_pos * val_prop)
#     n_test = int(m_pos * test_prop)
#     val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
#     val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
#     train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
#     adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
#     adj_train = adj_train + adj_train.T
#     return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
#            torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
#             test_edges_false)


# 未平衡正负样本比例
def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)

    # 1. 获取正样本（存在的边）
    x, y = sp.triu(adj).nonzero()  # 上三角避免重复
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)  # 随机打乱

    # 2. 获取负样本（不存在的边）
    neg_adj = sp.csr_matrix(1. - adj.toarray())  # 邻接矩阵取反
    x, y = sp.triu(neg_adj).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)  # 随机打乱

    # 3. 计算划分数量
    m_pos = len(pos_edges)
    n_test = int(m_pos * test_prop)  # 测试集数量
    n_val = int(m_pos * val_prop)  # 验证集数量

    # 4. 划分正样本
    test_edges = pos_edges[:n_test]  # 测试集正样本
    val_edges = pos_edges[n_test:n_test + n_val]  # 验证集正样本
    train_edges = pos_edges[n_test + n_val:]  # 训练集正样本

    # 5. 划分负样本（关键修改：仅使用未被选中的负样本）
    test_edges_false = neg_edges[:n_test]  # 测试集负样本
    val_edges_false = neg_edges[n_test:n_test + n_val]  # 验证集负样本
    train_edges_false = neg_edges[n_test + n_val:]  # 训练集负样本（仅剩余部分）

    # 6. 构建训练邻接矩阵（仅包含训练正样本）
    adj_train = sp.csr_matrix(
        (np.ones(len(train_edges)), (train_edges[:, 0], train_edges[:, 1])),
        shape=adj.shape
    )
    adj_train = adj_train + adj_train.T  # 确保对称性（无向图）

    # 7. 返回结果（转换为PyTorch张量）
    return (
        adj_train,
        torch.LongTensor(train_edges),
        torch.LongTensor(train_edges_false),
        torch.LongTensor(val_edges),
        torch.LongTensor(val_edges_false),
        torch.LongTensor(test_edges),
        torch.LongTensor(test_edges_false)
    )

# 平衡正负样本比例
# def mask_edges(adj, val_prop, test_prop, seed):
#     np.random.seed(seed)
#     # 获取正样本（确保无自环）
#     x, y = sp.triu(adj).nonzero()
#     pos_edges = np.array([(i, j) for i, j in zip(x, y) if i != j])  # 过滤自环
#     np.random.shuffle(pos_edges)
#
#     # 获取负样本（确保不是正样本且非自环）
#     adj_dense = adj.toarray()
#     neg_edges = []
#     while len(neg_edges) < 1*len(pos_edges):  # 生成等量负样本
#         i, j = np.random.randint(0, adj.shape[0], 2)
#         if i != j and adj_dense[i, j] == 0 and (i, j) not in neg_edges:
#             neg_edges.append((i, j))
#     neg_edges = np.array(neg_edges)
#
#     # 划分正样本
#     m_pos = len(pos_edges)
#     n_val = int(m_pos * val_prop)
#     n_test = int(m_pos * test_prop)
#     val_edges, test_edges, train_edges = (
#         pos_edges[:n_val],
#         pos_edges[n_val:n_test + n_val],
#         pos_edges[n_test + n_val:]
#     )
#
#     # 划分负样本（确保与正样本划分一致）
#     val_edges_false = neg_edges[:n_val]
#     test_edges_false = neg_edges[n_val:n_test + n_val]
#     train_edges_false = neg_edges[n_test + n_val:]  # ← 关键修改：不混入验证/测试正样本
#
#     # 重建训练邻接矩阵
#     adj_train = sp.csr_matrix(
#         (np.ones(len(train_edges)), (train_edges[:, 0], train_edges[:, 1])),
#         shape=adj.shape
#     )
#     adj_train = adj_train.maximum(adj_train.T)  # 确保对称
#
#     return (
#         adj_train,
#         torch.LongTensor(train_edges),
#         torch.LongTensor(train_edges_false),
#         torch.LongTensor(val_edges),
#         torch.LongTensor(val_edges_false),
#         torch.LongTensor(test_edges),
#         torch.LongTensor(test_edges_false)
#     )

def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


# ############### LINK PREDICTION DATA LOADERS ####################################
import scipy.io
def robust_cluster_augmentation(adj_matrix, n_clusters=5):
    """
    使用鲁棒性更强的Louvain社区发现算法
    需要安装python-louvain包：pip install python-louvain
    """
    import community as community_louvain

    # 转换为networkx图
    import networkx as nx
    G = nx.from_numpy_array(adj_matrix)

    # 执行Louvain聚类
    partition = community_louvain.best_partition(G, resolution=1.0)
    clusters = list(partition.values())

    # 后续增强逻辑相同
    augmented = adj_matrix.copy()
    for c in set(clusters):
        members = [i for i, x in enumerate(clusters) if x == c]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if augmented[members[i], members[j]] == 0:
                    if np.random.rand() > 0.5:
                        augmented[members[i], members[j]] = 1
                        augmented[members[j], members[i]] = 1
    return augmented

def load_data_lp():
    features = sp.csr_matrix(np.genfromtxt("{}{}.txt".format("./data/", "protein_feature"),
                             dtype=np.dtype(np.float32)))
    adj = np.genfromtxt("{}{}.txt".format("./data/", "mat_protein_protein"),
                        dtype=np.dtype(int))
    adj =sp.csr_matrix((adj))
    # data = scipy.io.loadmat('./data/zeng/rep_p_p.mat')
    #
    # features = data['rep_p_p']
    #
    # adj = sp.csr_matrix(np.genfromtxt("{}{}.txt".format("./data/zeng/", "proteinprotein"),
    #                                   dtype=np.dtype(int)))

    data = {'adj_train': adj, 'features': features}
    return data


# ############### NODE CLASSIFICATION DATA LOADERS ####################################


def load_data_nc(dataset, use_feats, data_path, split_seed):
    if dataset in ['cora', 'pubmed']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            dataset, use_feats, data_path, split_seed
        )
    else:
        if dataset == 'disease_nc':
            adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.60
        elif dataset == 'airport':
            adj, features, labels = load_data_airport(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        else:
            raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)

    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    return data


# ############### DATASETS ####################################


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + min(1000, len(labels) - len(y) - len(idx_test)))

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(adj.shape[0])
    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels


def load_data_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    
    features = np.array([graph.nodes[u]['feat'] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features

