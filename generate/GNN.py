import numpy as np
import torch
import dgl
import dgl.nn.pytorch as dglnn
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F



def create_hg(A, R, B, EOT, device):
    hetero_graph = dgl.heterograph({
    # connect A -> R
    ('object', 'link_or', 'relation'): ([0], [0]),
    # connect R -> B
    ('relation', 'link_ro', 'object'): ([0], [1]),
    # A -> EOT, B -> EOT
    ('object', 'link_oe', 'eot'): ([0 ,1], [0, 0]),
    # R -> EOT
    ('relation', 'link_re', 'eot'): ([0], [0])})
    hetero_graph = hetero_graph.to(device)
    hetero_graph.nodes['object'].data['feature'] = torch.concat([A, B])
    hetero_graph.nodes['relation'].data['feature'] = R
    hetero_graph.nodes['eot'].data['feature'] = EOT
    
    obj_features = torch.concat([A, B])
    R_features = R
    EOT_features = EOT
    
    return hetero_graph, {"object":obj_features, "relation":R_features, "eot":EOT_features}


def create_hg_self_loop(A, R, B, EOT, device):
    hetero_graph = dgl.heterograph({
    # connect A -> R
    ('object', 'link_or', 'relation'): ([0], [0]),
    # add self loop
    ('object', 'link_oo', 'object'): ([0, 1], [0, 1]),
    # connect R -> B
    ('relation', 'link_ro', 'object'): ([0], [1]),
    # A -> EOT, B -> EOT
    ('object', 'link_oe', 'eot'): ([0 ,1], [0, 0]),
    # R -> EOT
    ('relation', 'link_re', 'eot'): ([0], [0])})
    hetero_graph = hetero_graph.to(device)
    hetero_graph.nodes['object'].data['feature'] = torch.concat([A, B])
    hetero_graph.nodes['relation'].data['feature'] = R
    hetero_graph.nodes['eot'].data['feature'] = EOT
    
    obj_features = torch.concat([A, B])
    R_features = R
    EOT_features = EOT
    
    return hetero_graph, {"object":obj_features, "relation":R_features, "eot":EOT_features}

def create_hg_A_ARB_self_loop(A, R, B, A_EOT, ARB_EOT, device):
    # if add_noise_on_A:
    #     noise = torch.randn_like(A_EOT)
    #     A_EOT = A_EOT + 0.01*noise
    # return A and ARB 
    # EOT: 2, 1024, first one is A's second one is ARB's
    # connect A -> R
    hetero_graph = dgl.heterograph({
    # add self loop
    # ('eot', 'link_ee', 'eot'): ([1], [1]),
    ('object', 'link_oo', 'object'): ([0, 1], [0, 1]),
    ('object', 'link_or', 'relation'): ([0], [0]),
    # connect R -> B
    ('relation', 'link_ro', 'object'): ([0], [1]),
    # A -> ARB EOT, B -> ARB EOT, A-> A EOT
    ('object', 'link_oe', 'eot'): ([0 ,1, 0], [1, 1, 0]),
    # R -> ARB EOT
    ('relation', 'link_re', 'eot'): ([0], [1])})
    hetero_graph = hetero_graph.to(device)
    hetero_graph.nodes['object'].data['feature'] = torch.concat([A, B])
    hetero_graph.nodes['relation'].data['feature'] = R
    hetero_graph.nodes['eot'].data['feature'] = torch.concat([A_EOT, ARB_EOT])
    
    obj_features = torch.concat([A, B])
    R_features = R
    EOT_features = torch.concat([A_EOT, ARB_EOT])
    
    return hetero_graph, {"object":obj_features, "relation":R_features, "eot":EOT_features}

def create_batch_hg_A_ARB_B_self_loop(A, R, B, A_EOT, ARB_EOT, B_EOT, device):  #[1, 768], [1, 768] , [1, 768]
    
    batch_size = A.shape[0]
    graphs = []
    features = []

    for i in range(batch_size):
        g = dgl.heterograph({
        # add self loop
        # ('eot', 'link_ee', 'eot'): ([1], [1]),
        ('object', 'link_oo', 'object'): ([0, 1], [0, 1]),
        ('object', 'link_or', 'relation'): ([0], [0]),
        # connect R -> B
        ('relation', 'link_ro', 'object'): ([0], [1]),
        # A -> ARB EOT, B -> ARB EOT, A-> A EOT , B-> B EOT
        ('object', 'link_oe', 'eot'): ([0 ,1, 0, 1], [1, 1, 0, 2]),
        # R -> ARB EOT
        ('relation', 'link_re', 'eot'): ([0], [1])})
        g = g.to(device)
        g.nodes['object'].data['feature'] = torch.concat([A[i], B[i]])
        g.nodes['relation'].data['feature'] = R[i]
        g.nodes['eot'].data['feature'] = torch.concat([A_EOT[i], ARB_EOT[i],B_EOT[i]])
        graphs.append(g)

        obj_features = torch.concat([A[i], B[i]])  #[2, 768]
        R_features = R[i]    #[1, 768]
        EOT_features = torch.concat([A_EOT[i], ARB_EOT[i],B_EOT[i]])  #[3, 768]
        
        feature = {"object":obj_features, "relation":R_features, "eot":EOT_features}
        features.append(feature)

    return graphs, features


def create_hg_A_ARB_B_self_loop(A, R, B, A_EOT, ARB_EOT, B_EOT, device):
    # if add_noise_on_A:
    #     noise = torch.randn_like(A_EOT)
    #     A_EOT = A_EOT + 0.01*noise
    # return A and ARB 
    # EOT: 2, 1024, first one is A's second one is ARB's
    # connect A -> R
    hetero_graph = dgl.heterograph({
    # add self loop
    # ('eot', 'link_ee', 'eot'): ([1], [1]),
    ('object', 'link_oo', 'object'): ([0, 1], [0, 1]),
    ('object', 'link_or', 'relation'): ([0], [0]),
    # connect R -> B
    ('relation', 'link_ro', 'object'): ([0], [1]),
    # A -> ARB EOT, B -> ARB EOT, A-> A EOT , B-> B EOT
    ('object', 'link_oe', 'eot'): ([0 ,1, 0, 1], [1, 1, 0, 2]),
    # R -> ARB EOT
    ('relation', 'link_re', 'eot'): ([0], [1])})
    hetero_graph = hetero_graph.to(device)
    hetero_graph.nodes['object'].data['feature'] = torch.concat([A, B])
    hetero_graph.nodes['relation'].data['feature'] = R
    hetero_graph.nodes['eot'].data['feature'] = torch.concat([A_EOT, ARB_EOT,B_EOT])
    
    obj_features = torch.concat([A, B])  #[2, 768]
    R_features = R    #[1, 768]
    EOT_features = torch.concat([A_EOT, ARB_EOT,B_EOT])  #[3, 768]
    
    return hetero_graph, {"object":obj_features, "relation":R_features, "eot":EOT_features}

def create_hg_ARB_self_loop(A, R, B, A_EOT, ARB_EOT, B_EOT, device):
    # if add_noise_on_A:
    #     noise = torch.randn_like(A_EOT)
    #     A_EOT = A_EOT + 0.01*noise
    # return A and ARB 
    # EOT: 2, 1024, first one is A's second one is ARB's
    # connect A -> R
    hetero_graph = dgl.heterograph({
    # add self loop
    # ('eot', 'link_ee', 'eot'): ([1], [1]),
    ('object', 'link_oo', 'object'): ([0, 1], [0, 1]),
    ('object', 'link_or', 'relation'): ([0], [0]),
    # connect R -> B
    ('relation', 'link_ro', 'object'): ([0], [1]),
    # A -> ARB EOT, B -> ARB EOT
    ('object', 'link_oe', 'eot'): ([0 ,1], [0, 0]),
    # R -> ARB EOT
    ('relation', 'link_re', 'eot'): ([0], [0])})
    hetero_graph = hetero_graph.to(device)
    hetero_graph.nodes['object'].data['feature'] = torch.concat([A, B])
    hetero_graph.nodes['relation'].data['feature'] = R
    hetero_graph.nodes['eot'].data['feature'] =  ARB_EOT
    
    obj_features = torch.concat([A, B])  #[2, 768]
    R_features = R    #[1, 768]
    EOT_features = ARB_EOT  #[3, 768]
    
    return hetero_graph, {"object":obj_features, "relation":R_features, "eot":EOT_features}

    

class RGAT(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, heads=[2,2]):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(in_feats, hid_feats, heads[0])
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(hid_feats* heads[0], out_feats, heads[1])
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v).flatten(1) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: v.mean(1) for k, v in h.items()}
        
        return h

class HeteroGraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, rel_names, aggregator_type='mean'):
        super(HeteroGraphSAGE, self).__init__()
        
        # 定义异构图卷积层，使用 SAGEConv 进行邻居聚合
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(in_feats, hidden_feats, aggregator_type)  # 每种关系类型的聚合方式
            for rel in rel_names
        }, aggregate='mean')  # 聚合方式，sum、mean 或其他
        
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(hidden_feats, out_feats, aggregator_type)  # 每种关系类型的聚合方式
            for rel in rel_names
        }, aggregate='mean')
    
    def forward(self, graph, inputs):
        # 图卷积操作
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}  # 激活函数
        h = self.conv2(graph, h)  # 第二层卷积
        #h = {k: v.mean(1) for k, v in h.items()}  # 聚合操作，取均值
        
        return h


class RGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, rel_names):
        super(RGCN, self).__init__()
        
        # 定义异构图卷积层
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hidden_feats)  # 每种关系类型的卷积层
            for rel in rel_names
        }, aggregate='sum')  # 聚合方式，sum、mean 或其他
        
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hidden_feats, out_feats)  # 每种关系类型的卷积层
            for rel in rel_names
        }, aggregate='sum')
    
    def forward(self, graph, inputs):
        # 图卷积操作
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}  # 激活函数
        h = self.conv2(graph, h)  # 第二层卷积
        
        return h