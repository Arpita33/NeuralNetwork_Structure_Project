import glob 
import pickle
import torch
import torch_geometric
from torch_geometric.utils import from_networkx
from torch_geometric.utils import negative_sampling
import networkx as nx
import numpy as np
from tqdm import tqdm
import pickle
from model_codes import *
from graph_encoder import *


mlp_checkpoints_path = "/home/arpita-local/nn_structure_project/mlp_checkpoints"
pyg_graphs_path = "/home/arpita-local/nn_structure_project/pyg_graphs"
vgae_model_path = "/home/arpita-local/nn_structure_project/vgae_models"

DEVICE = "cuda:4" if torch.cuda.is_available() else "cpu"
print(DEVICE)
current_model_name = "dense"

model_names = ["sparse_deep", "sparse_deep2", "sparse_wide", "dense"]
for m in model_names:
    if not os.path.exists(f"{pyg_graphs_path}/{m}"): 
        os.makedirs(f"{pyg_graphs_path}/{m}")

if not os.path.exists(vgae_model_path): 
    os.makedirs(vgae_model_path)

def extract_number(name):
    # Extracting digits from the string and converting them to integer
    number = int(''.join(filter(str.isdigit, name))) #checkpoint -> epoch
    return number

def convert_weights_to_adjacency(weights):
    """
    converts weights of MLP to an adjacency matrix
    """
    #print(weights.shape)
    total_nodes = 0
    row, column = 0, 0
    input_nopdes = weights[0].shape[1]
    total_nodes += input_nopdes
    for weight in weights:
        total_nodes += weight.shape[0]
        
    adj_matrix = np.zeros((total_nodes, total_nodes))
    for weight in weights:
        dim0, dim1 = weight.shape
        row += dim1
        adj_matrix[row:row+dim0, column:column+dim1] = weight
        column += dim1
        
    return adj_matrix

def get_weight(path):
    print(path)

    if "sparse_deep2" in path:
        model = MLP_sparse_deep2()
    elif "sparse_deep" in path:
        model = MLP_sparse_deep()
    elif "sparse_wide" in path:
        model = MLP_sparse_wide()

    model.load_state_dict(torch.load(path))
    print("buffers")
    print(model.buffers())

    weights = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Do something with the name and the parameter tensor (e.g., append to weights)
            print(name)
            param=param.detach().numpy() 
            weights.append(param)

    # weights=weights
    return weights

def get_weight_sparse(path):
    #print(path)

    if "sparse_deep2" in path:
        model = MLP_sparse_deep2()
    elif "sparse_deep" in path:
        model = MLP_sparse_deep()
    elif "sparse_wide" in path:
        model = MLP_sparse_wide(
    elif "dense" in path:
        model = MLP_dense()
    
    #print(path)
    model.load_state_dict(torch.load(path))

    weights = []
    # print(model)

    with torch.no_grad():
        for layer in model.children():    
            if isinstance(layer, sl.SparseLinear):
                #print(layer.weight.shape)
                w = layer.weight.detach()
                w= w.to_dense().numpy()
                weights.append(w)
                # print(layer.weight.shape)

    # weights=weights
    return weights

def get_node_features(node_count: int):
    feature_dim = 1
    loc = np.random.uniform(0, 1)
    scale = np.random.uniform(0, 1)
    feature_vec = np.random.normal(loc, scale, size=(node_count, feature_dim))
    feature_tensor = torch.Tensor(feature_vec)
    return feature_tensor


def get_pyg_graphs(nx_graph: nx.DiGraph):
    g = from_networkx(nx_graph)
    node_count = len(nx_graph.nodes())
    g.x = get_node_features(node_count)
    g.edge_attr = g.weight
    del g.weight
    return g

def process_checkpoint(ckpt):
    # epoch_number=int(ckpt.split('_')[1].split('.')[0])
    weights=get_weight_sparse(ckpt)
    layer_count = 0
    weights_without_biases= []
    for layer in weights:
        if(layer_count%2==0):
            # print(layer.shape)
            weights_without_biases.append(layer)

        layer_count+=1

    adj_matrix = convert_weights_to_adjacency(weights_without_biases)
    graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    pg_graph = get_pyg_graphs(graph)
    epoch = int(ckpt.split("ckpt")[1].split("_")[1].split(".")[0])
    path_part = ckpt.split('.')[0]
    parts = path_part.split('/')
    path_part = parts[-2]+"_"+parts[-1]
    #print(path_part)
    # Use the pre-computed parts in your f-string without needing to include backslashes inside the expression braces
    #formatted_path = f"/home/arpita-local/pyg_graphs/{path_part}-{epoch}.pkl"
    formatted_path = f"/{pyg_graphs_path}/{current_model_name}/{path_part}-{epoch}.pkl"

    with open(formatted_path, 'wb') as f:
       pickle.dump(pg_graph, f)

def custom_train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1):
    # Ensure edge_index is coalesced for unique edges
    data.edge_index, _ = torch_geometric.utils.remove_self_loops(data.edge_index)
    data.edge_index, _ = torch_geometric.utils.coalesce(data.edge_index, None, data.num_nodes, data.num_nodes)

    # Shuffle edges
    all_edges = data.edge_index.t().cpu().numpy()
    np.random.shuffle(all_edges)
    
    # Determine the number of validation and test edges
    num_edges = all_edges.shape[0]
    num_val = int(num_edges * val_ratio)
    num_test = int(num_edges * test_ratio)
    
    # Split edges
    val_edges = all_edges[:num_val]
    test_edges = all_edges[num_val:num_val+num_test]
    train_edges = all_edges[num_val+num_test:]
    
    # Convert numpy arrays back to torch tensors
    data.train_pos_edge_index = torch.tensor(train_edges, dtype=torch.long).t().contiguous().to(data.edge_index.device)
    data.val_pos_edge_index = torch.tensor(val_edges, dtype=torch.long).t().contiguous().to(data.edge_index.device)
    data.test_pos_edge_index = torch.tensor(test_edges, dtype=torch.long).t().contiguous().to(data.edge_index.device)
    
    # Create negative samples for validation and testing
    data.val_neg_edge_index = negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes,
        num_neg_samples=val_edges.shape[0], method='sparse').to(data.edge_index.device)
    
    data.test_neg_edge_index = negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes,
        num_neg_samples=test_edges.shape[0], method='sparse').to(data.edge_index.device)

    # Extract edge attributes for train/val/test edges if edge_attr exists
    if 'edge_attr' in data:
        edge_attr = data.edge_attr.cpu().numpy()
        data.train_edge_attr = torch.tensor(edge_attr[num_val+num_test:], dtype=torch.float).to(data.edge_index.device)
        data.val_edge_attr = torch.tensor(edge_attr[:num_val], dtype=torch.float).to(data.edge_index.device)
        data.test_edge_attr = torch.tensor(edge_attr[num_val:num_val+num_test], dtype=torch.float).to(data.edge_index.device)
    
    return data



print(current_model_name)
model_paths = []
#model_folders = glob.glob('/home/arpita-local/mlp_checkpoints/*')
model_folders = glob.glob(f'{mlp_checkpoints_path}/{current_model_name}')

for model in model_folders:
    files = glob.glob(model + '/*.pth')
    files = sorted(files, key=extract_number)
    for file in files:
        model_paths.append(file)

performances_test = []
performances_train = []

for model_path in model_folders:
    val_acc_list_path =  model_path + '/validation_acc.pkl'
    with open(val_acc_list_path, 'rb') as f:
        val_acc_list = pickle.load(f)
    for val_acc in val_acc_list:
        performances_test.append(val_acc)
    
    train_acc_list_path =  model_path + '/training_acc.pkl'
    with open(train_acc_list_path, 'rb') as f:
        train_acc_list = pickle.load(f)
    for train_acc in train_acc_list:
        performances_train.append(train_acc)

#convert checkpoints to pyg graphs
for path in tqdm(model_paths):
    process_checkpoint(path)

def train_vgae(model, datas, optimizer):
    model.to(DEVICE)
    total_loss = 0
    total_ap, total_auc = 0, 0
    #print(datas)
    for data in tqdm(datas):
        
        #path = f"/home/arpita-local/pyg_graphs/{data}"
        path = f"{pyg_graphs_path}/{current_model_name}/{data}"
        g_data = pickle.load(open(path, "rb")).to(DEVICE)
        g_data.edge_attr=g_data.edge_attr.view(-1,1)      
        g_data = custom_train_test_split_edges(g_data)
        loss = train(model, optimizer, g_data)
        total_loss += loss
        auc, ap = test(model, g_data)
        total_auc += auc
        total_ap += ap
        
    return total_loss / len(datas), total_ap / len(datas), total_auc / len(datas)

# if __name__ == "__main__":
graphs_root = f"{pyg_graphs_path}/{current_model_name}"
graphs_list = os.listdir(graphs_root)
#print(graphs_list)

# graphs_list = [graph for graph in graphs_list if "conv" in graph]
# graphs_list = sorted(graphs_list, key=extract_number)

# set seed
torch.manual_seed(0)
    
    
graph_encoder = VariationalTransformerEncoder(1, 16,32,1)
# graph_encoder = VariationalGATEncoder(128, 16, 32)
model = VGAE(graph_encoder).to(DEVICE)
# load model state dictionary 
# model.load_state_dict(torch.load("vgae_models/best_autoencoder_conv.pth"))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
#epochs = 30
epochs = 200
best_auc = 0
losses=[]
aucs=[]
for i in (range(epochs)):
    loss, mean_ap, mean_auc = train_vgae(model, graphs_list, optimizer)
    losses.append(loss)
    aucs.append(mean_auc)
    print(mean_auc)
    print(f"Epoch {i+1}/{epochs}, Loss: {loss}")
    if mean_auc > best_auc:
        torch.save(
            model.state_dict(),
            f"{vgae_model_path}/best_autoencoder_{current_model_name}_{i}.pth"
        )
        best_auc = mean_auc

