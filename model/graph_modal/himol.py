"""
SciBert + GNN
"""

import torch
from torch import nn
from torch_geometric.data import Batch
import torch.nn.functional as F

import torch
import torch.nn.utils.rnn as rnn_utils
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax, remove_self_loops
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter
from helper_utils import NestedTensor
from typing import Any

import time

num_atom_type = 121  
num_chirality_tag = 11  

num_bond_type = 7  
num_bond_direction = 3


class GINConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self,
                 emb_dim,
                 heads=2,
                 negative_slope=0.2,
                 dropout=0.,
                 bias=True):
        super(GATConv, self).__init__(node_dim=0, aggr='add')  

        self.in_channels = emb_dim
        self.out_channels = emb_dim
        self.edge_dim = emb_dim  
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.weight = Parameter(torch.Tensor(emb_dim, heads * emb_dim))  
        self.att = Parameter(
            torch.Tensor(1, heads, 2 * emb_dim + self.edge_dim))  
        self.edge_update = Parameter(
            torch.Tensor(emb_dim + self.edge_dim, emb_dim))  

        if bias:
            self.bias = Parameter(torch.Tensor(emb_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        glorot(self.edge_update)  
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        edge_attr = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)

        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        self_loop_edges = torch.zeros(x.size(0), edge_attr.size(1)).to(edge_index.device)
        edge_attr = torch.cat([edge_attr, self_loop_edges], dim=0)

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

    def message(self, x_i, x_j, size_i, edge_index_i, edge_attr):

        edge_attr = edge_attr.unsqueeze(1).repeat(1, self.heads, 1)
        x_j = torch.cat([x_j, edge_attr], dim=-1)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = torch.mm(aggr_out, self.edge_update)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GNN(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, hidden_size, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer  
        self.drop_ratio = drop_ratio  
        self.JK = JK  

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, hidden_size)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, hidden_size)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(hidden_size, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(hidden_size))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(hidden_size))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(hidden_size))

        
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_size))

    def forward(self, *argv) -> torch.Tensor:
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        
        h_list = x.unsqueeze(0)
        for layer in range(self.num_layer):

            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            
            h_list = torch.cat((h_list, h.unsqueeze(0)), dim=0)

        
        
        if self.JK == "concat":
            node_representation = h_list.permute(1, 0, 2).flatten(1)
        elif self.JK == "last":  
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation


class HiMolGraphTower(nn.Module):
    def __init__(
        self,
        encoder_num_layer: int,  
        hidden_size: int,  
        encoder_JK: str,
        encoder_drop_ratio: float,
        encoder_gnn_type: str,
    ) -> None:
        super().__init__()
        DeprecationWarning(f"class {self.__class__} is deprecated, try HiMolGraphTowerV2")
        self.hidden_size = hidden_size
        self.encoder = GNN(
            num_layer=encoder_num_layer,
            hidden_size=hidden_size,  
            JK=encoder_JK,
            drop_ratio=encoder_drop_ratio,
            gnn_type=encoder_gnn_type
        )
        
    @torch.inference_mode()
    def group_node_rep(
            self,
            node_rep: torch.Tensor,
            batch_size: int,
            num_part: int
    ) -> list[list[torch.Tensor]]:
        group = []
        count = 0
        
        for i in range(batch_size):
            collected_group = []
            num_atom = num_part[i][0]
            num_motif = num_part[i][1]
            num_all = num_atom + num_motif + 1
            
            collected_group.append(node_rep[count:count + num_atom])  
            if num_motif == 0:
                
                collected_group.append(torch.full((1, 300), float('nan')))  
                
            else:
                collected_group.append(node_rep[count + num_atom:count + num_atom + num_motif])  
            collected_group.append(node_rep[count + num_all - 1].unsqueeze(0))

            group.append(collected_group)
            
            count += num_all

        return group
    
    @torch.inference_mode()
    def pad_nested_sequences(self, data: list[list[torch.Tensor]], device='cuda'):
        batch_size, num_levels = len(data), len(data[0])
        
        padded_levels = []
        masks = []
        dtype = data[0][0].dtype
        
        max_length = max(len(item) for batch in data for item in batch)
        
        
        for level in range(num_levels):
            mask = torch.ones((batch_size, max_length), dtype=torch.bool, device=device)
            
            level_data = [data[batch][level] for batch in range(batch_size)]  
            
            padded_level = rnn_utils.pad_sequence(level_data, batch_first=True, padding_value=0.0).to(dtype).to(device)
            
            
            if padded_level.size(1) < max_length:
                
                extra_padding = max_length - padded_level.size(1)  
                
                
                padding = torch.zeros((batch_size, extra_padding, padded_level.size(2)), dtype=dtype, device=device)
                padded_level = torch.cat([padded_level, padding], dim=1)
                
            
            
            for batch in range(batch_size):
                if torch.isnan(padded_level[batch]).any():
                    
                    padded_level[batch] = torch.zeros((1, 300), dtype=dtype, device=device)
                    mask[batch, :] = True  
                else:
                    mask[batch, :data[batch][level].shape[0]] = False

            masks.append(mask)
            padded_levels.append(padded_level)
        
        padded_data = torch.stack(padded_levels, dim=1)
        maskings = torch.stack(masks, dim=1)
        
        return NestedTensor(padded_data, maskings)
    

    def load_weight(self, init_ckpt: str):
        print("load weight from", init_ckpt)
        weight = torch.load(init_ckpt)
        weight = {'encoder.'+k:v for k, v in weight.items()}  
        self.load_state_dict(weight)

    
    @torch.inference_mode()
    def forward(
            self,
            graph_batch: Batch,
            batch_size: int
    ) -> NestedTensor:
        
        graph = graph_batch
        

        
        node_rep = self.encoder(
            graph.x,
            graph.edge_index,
            graph.edge_attr
        )  

        num_part = graph.num_part
        node_repr = self.group_node_rep(node_rep, batch_size, num_part)
        
        nest_tensor = self.pad_nested_sequences(node_repr)  
        

        return nest_tensor


class GraphFeature():
    def __init__(self, data: list[list[torch.Tensor]], device: torch.device) -> None:
        """A helpful class for pre-processing graph feature

        Args:
            data (list[list[torch.Tensor]]): input grouped graph feature
            device (torch.device): device of the data
        """
        self.data = data
        self.device = device
        
    def pad_to(self, shape:str) -> NestedTensor:
        if shape == "blnd":
            return self.pad_to_blnd()
        elif shape == "bnd":
            return self.pad_to_bnd()
        else:
            raise ValueError(f"Unknow shape {shape}")
        
    def to(self, device):
        self.device = device
        
    @torch.inference_mode
    def pad_to_blnd(self) -> NestedTensor:
        """Pad graph feature to Nested Tensor with shape B x Level x N x D

        Returns:
            NestedTensor: Padded graph feature
        """
        
        B = len(self.data)
        L = len(self.data[0])

        
        max_N = max(tensor.shape[0] for sample in self.data for tensor in sample)
        D = self.data[0][0].shape[1]

        
        padded_tensors = torch.zeros((B, L, max_N, D), dtype=self.data[0][0].dtype, device=self.device)
        mask = torch.ones((B, L, max_N), dtype=torch.bool, device=self.device)

        
        for b_idx, levels in enumerate(self.data):
            for l_idx, tensor in enumerate(levels):
                N = tensor.shape[0]
                
                
                if torch.isnan(tensor).any():
                    continue
                padded_tensors[b_idx, l_idx, :N] = tensor
                mask[b_idx, l_idx, :N] = False
        
        return NestedTensor(padded_tensors, mask).to(self.device)
        
    
    def pad_to_bnd(self) -> NestedTensor:
        """Pad graph feature to Nested Tensor with shape B x Level*N x D

        Returns:
            NestedTensor: Padded feature
        """
        
        total_batch = []
        len_of_tensors = []
        
        for batch in self.data:
            
            batch_filtered = [feature for feature in batch if not torch.isnan(feature).any()]
            
            concatenated = torch.cat(batch_filtered, dim=0)
            total_batch.append(concatenated)
            len_of_tensors.append(concatenated.shape[0])

        
        total_batch = rnn_utils.pad_sequence(total_batch, batch_first=True, padding_value=0.0)

        
        B, N, D = total_batch.shape
        mask = torch.arange(N, device=self.device).expand(B, N) >= torch.tensor(len_of_tensors, device=self.device).unsqueeze(1)

        return NestedTensor(total_batch, mask).to(self.device)
        
    
    def get_level(self, level: str, pad: bool) -> list[torch.Tensor] | NestedTensor:
        """Get one level of the feature

        Args:
            level (str): choose from node, motif or graph. Should notice that motif feature might not exists, the function returns torch.zeros
            pad (bool): whether to pad the feature across batches

        Returns:
            list[torch.Tensor] | NestedTensor: if padded, return nested tensor, otherwise a list of tensors
        """
        level_idx = {"node": 0, "motif": 1, "graph": 2}[level]
        collected_batch = []
        len_of_tensors = []
        dtype = self.data[0][0].dtype
        for batch in self.data:
            
            feature = batch[level_idx]
            if torch.isnan(feature).any():
                feature = torch.zeros((1, 300), dtype=dtype, device=self.device)
                
            collected_batch.append(feature)
            len_of_tensors.append(feature.shape[0])
            
        if pad:
            collected_batch = rnn_utils.pad_sequence(collected_batch, batch_first=True, padding_value=0.0)
            B, N, D = collected_batch.shape
            mask = torch.arange(N, device=self.device).expand(B, N) >= torch.tensor(len_of_tensors, device=self.device).unsqueeze(1)
                
            return NestedTensor(collected_batch, mask).to(self.device)
        else:
            return collected_batch
        


class HiMolGraphTowerV2(nn.Module):
    def __init__(
        self,
        encoder_num_layer: int,
        hidden_size: int,
        encoder_JK: str,
        encoder_drop_ratio: float,
        encoder_gnn_type: str,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = GNN(
            num_layer=encoder_num_layer,
            hidden_size=hidden_size,  
            JK=encoder_JK,
            drop_ratio=encoder_drop_ratio,
            gnn_type=encoder_gnn_type
        )
        
    @torch.inference_mode()
    def group_node_rep(
        self,
        node_rep: torch.Tensor,
        batch_size: int,
        num_part: int
    ) -> list[list[torch.Tensor]]:
        group = []
        count = 0
        
        for i in range(batch_size):
            num_atom, num_motif = num_part[i][0], num_part[i][1]
            num_all = num_atom + num_motif + 1

            atom_group = node_rep[count:count + num_atom]
            motif_group = node_rep[count + num_atom:count + num_atom + num_motif] \
            if num_motif > 0 else torch.full((1, 300), float('nan'))
            
            global_rep = node_rep[count + num_all - 1].unsqueeze(0)

            group.append([atom_group, motif_group, global_rep])

            count += num_all  

        return group

    def load_weight(self, init_ckpt: str):
        print("load weight from", init_ckpt)
        weight = torch.load(init_ckpt)
        weight = {'encoder.'+k:v for k, v in weight.items()}  
        self.load_state_dict(weight)

    @torch.inference_mode()
    def forward(self, graph: Batch, batch_size: int) -> GraphFeature:
        """Forward pass of HiMol graph encoder

        Args:
            graph (Batch): torch_geometric.data.Batch, graph data.
            batch_size (int): batch size

        Returns:
            GraphFeature: A useful class that can pad, or obtain feature from a certain level
        """
        node_rep = self.encoder(graph.x, graph.edge_index, graph.edge_attr)
        num_part = graph.num_part
        
        node_repr = self.group_node_rep(node_rep, batch_size, num_part)
        node_repr = GraphFeature(node_repr, node_repr[0][0].device)

        return node_repr
