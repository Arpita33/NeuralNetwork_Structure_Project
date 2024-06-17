import sparselinear as sl
import torch.nn as nn

# Define the MLP dense model (sparsity = 0)
class MLP_dense(nn.Module):
    def __init__(self):
        super(MLP_dense, self).__init__()
                #give name to each layer
        self.fc1 = sl.SparseLinear(1024, 1000, sparsity=0)
        self.fc2 = sl.SparseLinear(1000, 200 , sparsity=0)
        self.fc3 = sl.SparseLinear(200, 300, sparsity=0)
        self.fc4 = sl.SparseLinear(300, 6, sparsity=0) 
        self.relu = nn.ReLU()


    def forward(self, x):
        results= {}
        # return self.layers(x)
        x = self.relu(self.fc1(x))
        results['fc1'] = x
        x = self.relu(self.fc2(x))
        results['fc2'] = x
        x = self.relu(self.fc3(x))
        results['fc3'] = x
        x = (self.fc4(x))
        results['fc4'] = x

        return x, results


# MLP sparse wide
class MLP_sparse_wide(nn.Module):
    def __init__(self):
        super(MLP_sparse_wide, self).__init__()
                #give name to each layer
                #The model has 674930 parameters
        # self.fc1 = sl.SparseLinear(1024, 1000, sparsity=0.5)
        # self.fc2 = sl.SparseLinear(1000, 200 , sparsity=0.4)
        # self.fc3 = sl.SparseLinear(200, 300, sparsity=0.33)
        # self.fc4 = sl.SparseLinear(300, 6, sparsity=0.32)  

        #The model has 674970 parameters
        self.fc1 = sl.SparseLinear(1024, 1000, sparsity=0.60)
        self.fc2 = sl.SparseLinear(1000, 400 , sparsity=0.64)
        self.fc3 = sl.SparseLinear(400, 600, sparsity=0.51)
        self.fc4 = sl.SparseLinear(600, 6, sparsity=0.51)
        self.relu = nn.ReLU()


    def forward(self, x):
        results= {}
        # return self.layers(x)
        x = self.relu(self.fc1(x))
        results['fc1'] = x
        x = self.relu(self.fc2(x))
        results['fc2'] = x
        x = self.relu(self.fc3(x))
        results['fc3'] = x
        x = (self.fc4(x))
        results['fc4'] = x

        return x, results
    
# MLP sparse deep
class MLP_sparse_deep(nn.Module):
    def __init__(self):
        super(MLP_sparse_deep, self).__init__()
        #give name to each layer
        #674928 parameters
        self.fc1 = sl.SparseLinear(1024, 1000, sparsity=0.74)
        self.fc2 = sl.SparseLinear(1000, 800 , sparsity=0.74)
        self.fc3 = sl.SparseLinear(800, 600, sparsity=0.7)
        self.fc4 = sl.SparseLinear(600, 300, sparsity=0.705)
        self.fc5 = sl.SparseLinear(300, 6, sparsity=0.51)
        self.relu = nn.ReLU()


    def forward(self, x):
        results= {}
        # return self.layers(x)
        x = self.relu(self.fc1(x))
        results['fc1'] = x
        x = self.relu(self.fc2(x))
        results['fc2'] = x
        x = self.relu(self.fc3(x))
        results['fc3'] = x
        x = (self.fc4(x))
        results['fc4'] = x
        x = (self.fc5(x))
        results['fc5'] = x        

        return x, results

class MLP_sparse_deep2(nn.Module):
    def __init__(self):
        super(MLP_sparse_deep2, self).__init__()
                #give name to each layer
                #674928 parameters
        self.fc1 = sl.SparseLinear(1024, 1000, sparsity=0.73)
        self.fc2 = sl.SparseLinear(1000, 800 , sparsity=0.72)
        self.fc3 = sl.SparseLinear(800, 400, sparsity=0.70)
        self.fc4 = sl.SparseLinear(400, 600, sparsity=0.70)
        #self.fc5 = sl.SparseLinear(500, 600, sparsity=0.74)
        self.fc5 = sl.SparseLinear(600, 6, sparsity=0.5)
        self.relu = nn.ReLU()


    def forward(self, x):
        results= {}
        # return self.layers(x)
        x = self.relu(self.fc1(x))
        results['fc1'] = x
        x = self.relu(self.fc2(x))
        results['fc2'] = x
        x = self.relu(self.fc3(x))
        results['fc3'] = x
        x = (self.fc4(x))
        results['fc4'] = x
        x = (self.fc5(x))
        results['fc5'] = x       

        return x, results
    