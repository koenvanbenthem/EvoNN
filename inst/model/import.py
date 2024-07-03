import os
import pandas as pd
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
from math import ceil
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import dense_diff_pool
from torch_geometric.nn import DenseSAGEConv as SAGEConv
from torch.nn.utils.rnn import pack_padded_sequence