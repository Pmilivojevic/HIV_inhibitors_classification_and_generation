import torch
import torch.nn.functional as F 
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling 
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

torch.manual_seed(42)

class BlockGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_heads, dropout_rate, edge_dim):
        super(BlockGNN, self).__init__()
        self.conv = TransformerConv(
            in_channels, 
            out_channels, 
            heads=n_heads, 
            dropout=dropout_rate, 
            edge_dim=edge_dim, 
            beta=True
        )
        self.linear = Linear(out_channels * n_heads, out_channels)
        self.bn = BatchNorm1d(out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr)
        x = torch.relu(self.linear(x))
        x = self.bn(x)
        
        return x

class MolGNN(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        super(MolGNN, self).__init__()
        
        # Extract hyperparameters
        embedding_size = model_params.model_embedding_size
        n_heads = model_params.model_attention_heads
        self.n_layers = model_params.model_layers
        dropout_rate = model_params.model_dropout_rate
        top_k_ratio = model_params.model_top_k_ratio
        self.top_k_every_n = model_params.model_top_k_every_n
        dense_neurons = model_params.model_dense_neurons
        edge_dim = model_params.model_edge_dim

        # Define layers
        self.blocks = ModuleList([
            BlockGNN(
                feature_size if i == 0 else embedding_size,
                embedding_size,
                n_heads,
                dropout_rate,
                edge_dim
            ) for i in range(self.n_layers + 1)
        ])  # +1 for initial layer
        
        self.pooling_layers = ModuleList([
            TopKPooling(
                embedding_size,
                ratio=top_k_ratio
            ) for _ in range((self.n_layers // self.top_k_every_n) + 1)
        ])

        # Linear layers
        self.linear1 = Linear(embedding_size * 2, dense_neurons)
        self.linear2 = Linear(dense_neurons, int(dense_neurons / 2))
        self.linear3 = Linear(int(dense_neurons / 2), 1)

    def forward(self, x, edge_attr, edge_index, batch_index):
        global_representation = []

        # Process through GNN blocks
        for i, block in enumerate(self.blocks):
            x = block(x, edge_index, edge_attr)
            
            # Apply pooling at specified intervals or at the last layer
            if i % self.top_k_every_n == 0 or i == self.n_layers:
                pool_idx = min(int(i / self.top_k_every_n), len(self.pooling_layers) - 1)
                
                x, edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[pool_idx](
                    x,
                    edge_index,
                    edge_attr,
                    batch_index
                )
                
                global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))

        # Aggregate representations
        x = sum(global_representation)

        # Output block
        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = torch.relu(self.linear2(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.linear3(x)

        return x
