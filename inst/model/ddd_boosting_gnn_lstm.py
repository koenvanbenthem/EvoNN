import os
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
from math import ceil
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import dense_diff_pool
from torch_geometric.nn import DenseSAGEConv as SAGEConv
from torch.nn.utils.rnn import pack_padded_sequence

# Set global variables
cap_norm_factor = 1000
diffpool_ratio = 0.25
dropout_ratio = 0.5
gcn_layer1_hidden_channels = 256
gcn_layer2_hidden_channels = 128
gcn_layer3_hidden_channels = 64
dnn_hidden_channels = 64
dnn_output_channels = 32
lstm_hidden_channels = 64
lstm_output_channels = 32
lstm_depth = 5
lin_layer1_hidden_channels = 64
lin_layer2_hidden_channels = 32
n_predicted_values = 3
max_nodes_limit = 2500
normalize_edge_length = False
normalize_graph_representation = False
global_pooling_method = "mean"

# Set max node for DDD_TES
model_max_node = 2147
max_brts_len = 1073


def create_dataset(tree_nd, tree_el, tree_st, tree_bt):
    # List to hold the Data objects
    pytorch_geometric_data_list = []

    # Ensure the DataFrame is of integer type and convert to a tensor
    edge_index_tensor = torch.tensor(tree_nd, dtype=torch.long)

    # Make sure the edge_index tensor is of size [2, num_edges]
    edge_index_tensor = edge_index_tensor.t().contiguous()

    # Determine the number of nodes
    num_nodes = edge_index_tensor.max().item() + 1

    edge_length_tensor = torch.tensor(tree_el, dtype=torch.float)

    stats_tensor = torch.tensor(tree_st, dtype=torch.float)

    brts_tensor = torch.tensor(tree_bt, dtype=torch.float)

    brts_length = torch.tensor([len(tree_bt)], dtype=torch.long)

    # Create a Data object with the edge index, number of nodes, and category value
    data = Data(x=edge_length_tensor,
                edge_index=edge_index_tensor,
                num_nodes=num_nodes,
                stats=stats_tensor,
                brts=brts_tensor,
                brts_len=brts_length)

    # Append the Data object to the list
    pytorch_geometric_data_list.append(data)

    # Exclude data with stat or brts == 0, as they are not ultrametric or binary trees
    filtered_data_list = [
        data for data in pytorch_geometric_data_list
        if data.stats.shape != torch.Size([1])
    ]

    return filtered_data_list


def estimation(model_path, tree_nd, tree_el, tree_st, tree_bt, scale):
    depth = 2
    # Create dataset from imported data
    current_dataset = create_dataset(tree_nd, tree_el, tree_st, tree_bt)

    class TreeData(InMemoryDataset):
        def __init__(self, root, data_list, transform=None, pre_transform=None):
            # Calculate the maximum length of brts across all graphs
            max_length = max_brts_len

            # Pad the brts attribute for each graph
            for data in data_list:
                pad_size = max_length - len(data.brts)
                data.brts = torch.cat([data.brts, data.brts.new_full((pad_size,), fill_value=0)], dim=0)

            super(TreeData, self).__init__(root, transform, pre_transform)
            self.data, self.slices = self.collate(data_list)

        def _download(self):
            pass  # No download required

        def _process(self):
            pass  # No processing required

    emp_dataset = TreeData(root=None, data_list=current_dataset, transform=T.ToDense(model_max_node))

    class GNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels,
                     normalize=False):
            super(GNN, self).__init__()

            self.convs = torch.nn.ModuleList()
            self.bns = torch.nn.ModuleList()

            for i in range(depth):
                first_index = 0
                last_index = depth - 1

                if depth == 1:
                    self.convs.append(SAGEConv(in_channels, out_channels, normalize))
                    self.bns.append(torch.nn.BatchNorm1d(out_channels))
                else:
                    if i == first_index:
                        self.convs.append(SAGEConv(in_channels, hidden_channels, normalize))
                        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
                    elif i == last_index:
                        self.convs.append(SAGEConv(hidden_channels, out_channels, normalize))
                        self.bns.append(torch.nn.BatchNorm1d(out_channels))
                    else:
                        self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize))
                        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        def forward(self, x, adj, mask=None):
            outputs = []  # Initialize a list to store outputs at each step
            for step in range(len(self.convs)):
                x = F.gelu(self.convs[step](x, adj, mask))
                x = torch.permute(x, (0, 2, 1))
                x = self.bns[step](x)
                x = torch.permute(x, (0, 2, 1))
                outputs.append(x)  # Store the current x

            x_concatenated = torch.cat(outputs, dim=-1)
            return x_concatenated

    # The LSTM model class with variable number of layers to process variable-length branch time sequences
    class LSTM(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, lstm_depth=1, dropout_rate=0.5):
            super(LSTM, self).__init__()

            self.lstm = torch.nn.LSTM(input_size=in_channels, hidden_size=hidden_channels, num_layers=lstm_depth,
                                      batch_first=True, dropout=dropout_ratio if lstm_depth > 1 else 0)
            self.dropout = torch.nn.Dropout(dropout_rate)  # Dropout layer after LSTM
            self.lin = torch.nn.Linear(hidden_channels, out_channels)
            self.readout = torch.nn.Linear(out_channels, n_predicted_values)

        def forward(self, x):
            out, (h_n, c_n) = self.lstm(x)

            # Unpack the sequences
            # x, _ = pad_packed_sequence(x, batch_first=True)

            # Get the final hidden state
            final_hidden_state = h_n[-1, :, :]

            x = F.gelu(final_hidden_state)

            x = self.dropout(x)  # Apply dropout after LSTM and activation
            x = self.lin(x)
            x = F.gelu(x)
            x = self.dropout(x)
            x = self.readout(x)
            return x

    # Differential pooling model class
    # Multimodal architecture with GNNs, LSTMs, and DNNs
    class DiffPool(torch.nn.Module):
        def __init__(self, verbose=False):
            super(DiffPool, self).__init__()
            self.verbose = verbose
            if self.verbose:
                print("Initializing DiffPool model...")

            # Read in augmented data
            self.graph_sizes = torch.tensor([], dtype=torch.long)

            if self.verbose:
                print("Graph sizes loaded...")

            # DiffPool Layer 1
            num_nodes1 = ceil(diffpool_ratio * model_max_node)
            self.gnn1_pool = GNN(emp_dataset.num_node_features, gcn_layer1_hidden_channels, num_nodes1)
            self.gnn1_embed = GNN(emp_dataset.num_node_features, gcn_layer1_hidden_channels,
                                  gcn_layer2_hidden_channels)

            # DiffPool Layer 2
            num_nodes2 = ceil(diffpool_ratio * num_nodes1)
            gnn1_out_channels = gcn_layer1_hidden_channels * (depth - 1) + gcn_layer2_hidden_channels
            self.gnn2_pool = GNN(gnn1_out_channels, gcn_layer2_hidden_channels, num_nodes2)
            self.gnn2_embed = GNN(gnn1_out_channels, gcn_layer2_hidden_channels, gcn_layer3_hidden_channels)

            # DiffPool Layer 3
            gnn2_out_channels = gcn_layer2_hidden_channels * (depth - 1) + gcn_layer3_hidden_channels
            self.gnn3_embed = GNN(gnn2_out_channels, gcn_layer3_hidden_channels, lin_layer1_hidden_channels)
            gnn3_out_channels = gcn_layer3_hidden_channels * (depth - 1) + lin_layer1_hidden_channels

            # Final Readout Layers
            self.lin1 = torch.nn.Linear(gnn3_out_channels, lin_layer2_hidden_channels)
            self.lin2 = torch.nn.Linear(lin_layer2_hidden_channels, n_predicted_values)

        def forward(self, x, adj, mask=None):
            # Forward pass through the DiffPool layer 1
            s = self.gnn1_pool(x, adj, mask)
            x = self.gnn1_embed(x, adj, mask)
            x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

            if self.verbose:
                print("DiffPool Layer 1 Completed...")

            # Forward pass through the DiffPool layer 2
            s = self.gnn2_pool(x, adj)
            x = self.gnn2_embed(x, adj)
            x, adj, l2, e2 = dense_diff_pool(x, adj, s)

            if self.verbose:
                print("DiffPool Layer 2 Completed...")

            # Forward pass through the DiffPool layer 3
            x = self.gnn3_embed(x, adj)

            if self.verbose:
                print("DiffPool Layer 3 Completed...")

            # Global pooling after the final GNN layer
            if global_pooling_method == "mean":
                x = x.mean(dim=1)
            elif global_pooling_method == "max":
                x = x.max(dim=1).values
            elif global_pooling_method == "sum":
                x = x.sum(dim=1)
            else:
                raise ValueError("Invalid global pooling method.")

            if self.verbose:
                print("Global Pooling Completed...")

            # Forward pass through the readout layers
            x = F.dropout(x, p=dropout_ratio, training=self.training)
            x = self.lin1(x)
            x = F.gelu(x)
            x = F.dropout(x, p=dropout_ratio, training=self.training)
            x = self.lin2(x)
            # x = F.relu(x)
            if self.verbose:
                print("Readout Layers Completed...")
                print("Forward Pass Completed...")
                print("Epoch Completed...")

            # Return the final output along with the DiffPool layer losses
            return x, l1 + l2, e1 + e2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating using {device}")

    # Load pre-trained GNN model state
    model_gnn = DiffPool()
    model_gnn = model_gnn.to(device)
    path_to_gnn_model = os.path.join(model_path, "DDD_FREE_TES_model_diffpool_2_gnn.pt")
    model_gnn.load_state_dict(torch.load(path_to_gnn_model, map_location=device))

    # Load pre_trained LSTM model state
    model_lstm = LSTM(in_channels=1, hidden_channels=lstm_hidden_channels,
                      out_channels=lstm_output_channels, lstm_depth=lstm_depth).to(device)
    model_lstm = model_lstm.to(device)
    path_to_lstm_model = os.path.join(model_path, "DDD_FREE_TES_gnn_2_model_lstm.pt")
    model_lstm.load_state_dict(torch.load(path_to_lstm_model, map_location=device))

    emp_loader = DenseDataLoader(emp_dataset, batch_size=1, shuffle=False)

    predictions_before_lstm = torch.tensor([], dtype=torch.float, device=device)
    predictions_after_lstm = torch.tensor([], dtype=torch.float, device=device)
    num_nodes_original = torch.tensor([], dtype=torch.long, device=device)

    with torch.no_grad():
        model_gnn.eval()
        model_lstm.eval()
        for data in emp_loader:
            data.to(device)
            predictions, _, _ = model_gnn(data.x, data.adj, data.mask)
            num_nodes_original = torch.cat((num_nodes_original, data.num_nodes), dim=0)
            predictions_before_lstm = torch.cat((predictions_before_lstm, predictions), dim=0)
            lengths_brts = torch.sum(data.brts != 0, dim=1).cpu().tolist()
            brts_cpu = data.brts.cpu()
            brts_cpu = brts_cpu.unsqueeze(-1)
            packed_brts = pack_padded_sequence(brts_cpu, lengths_brts, batch_first=True, enforce_sorted=False).to(
                device)
            predicted_residual = model_lstm(packed_brts)
            predictions_after_lstm = torch.cat((predictions_after_lstm, predictions + predicted_residual), dim=0)

    # Save the data for the LSTM compensation
    predictions_before_lstm = predictions_before_lstm.cpu().detach().numpy()
    predictions_after_lstm = predictions_after_lstm.cpu().detach().numpy()
    num_nodes_original = num_nodes_original.cpu().detach().numpy()

    emp_data_dict = {"pred_lambda": predictions_after_lstm[:, 0],
                       "pred_mu": predictions_after_lstm[:, 1],
                       "pred_cap": predictions_after_lstm[:, 2],
                       "scale": scale,
                       "num_nodes": num_nodes_original}

    # Multiply the cap results by the normalization factor
    emp_data_dict["pred_cap"] *= cap_norm_factor

    # Convert cap predictions to the closest integer
    emp_data_dict["pred_cap"] = emp_data_dict["pred_cap"].round()

    return emp_data_dict