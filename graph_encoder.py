from typing import Callable, Optional
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Dataset
from torch_geometric.nn import VGAE, GATConv, GCNConv, GAE
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import TransformerConv

from typing import Any, List
import os
import os.path as osp
from copy import copy
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import pickle

#empty cuda cache
torch.cuda.empty_cache()

class graph_dataset(Dataset):
    def __init__(self, 
                 files_list: List[str],
                 root: str | None = None, 
                 transform: Callable[..., Any] | None = None, 
                 pre_transform: Callable[..., Any] | None = None, 
                 pre_filter: Callable[..., Any] | None = None, 
                 log: bool = True):
        super().__init__(root, transform, pre_transform, pre_filter, log)
        self.files_list = files_list
        
        
    @property
    def raw_file_names(self):
        return self.files_list
    
    @property
    def processed_file_names(self):
        return self.files_list
    
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
    

    
    
class VariationalTransformerEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, edge_dim):
        super(VariationalTransformerEncoder, self).__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, edge_dim=edge_dim, heads=1, dropout=0.2, concat=True)
        self.conv_mu = TransformerConv(hidden_channels * 1, out_channels, edge_dim=edge_dim, heads=1, dropout=0.2, concat=False)
        self.conv_logstd = TransformerConv(hidden_channels * 1, out_channels, edge_dim=edge_dim, heads=1, dropout=0.2, concat=False)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        mu = self.conv_mu(x, edge_index, edge_attr)
        logstd = self.conv_logstd(x, edge_index, edge_attr)
        return mu, logstd

    
    
# Example modification for the train function
def train(model, optimizer, g_data):
    model.train()
    optimizer.zero_grad()
    # print(g_data.x.shape)
    # print(g_data.train_pos_edge_index.shape)
    # print(g_data.edge_attr.shape)
    # print(g_data)
    z = model.encode(g_data.x, g_data.train_pos_edge_index,g_data.train_edge_attr)  # Include edge_attr
    loss = model.recon_loss(z, g_data.train_pos_edge_index)
    loss += (1 / g_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)

def fill_nodes_with_random_features(data, size):
    #set seed
    torch.manual_seed(0)
    data.x = torch.zeros(data.num_nodes, size)
    return data

def fill_edges_with_random_features(data, size):
    #set seed
    torch.manual_seed(0)
    data.weight = torch.zeros(data.num_edges, size)
    return data

def test(model, g_data):
    model.eval()
    with torch.no_grad():
        z = model.encode(g_data.x, g_data.train_pos_edge_index, g_data.train_edge_attr)  # Include edge_attr
    return model.test(z, g_data.test_pos_edge_index, g_data.test_neg_edge_index)  # Include edge_attr


