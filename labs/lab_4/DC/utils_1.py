import time, math
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict


# ================ MISC ================
def get_device(device_idx=None):
    if torch.cuda.is_available():
       if device_idx: device = device_idx
       else: device = torch.cuda.current_device()
       print("GPU available.")
    else:
       device=torch.device('cpu')
       print("GPU not available, falling back to CPU.")   
       
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    reserved_memory = torch.cuda.memory_reserved(device)
    print(f"Total memory: {total_memory / (1024 ** 3):.2f} GB")
    print(f"Allocated memory: {allocated_memory / (1024 ** 3):.2f} GB")
    print(f"Reserved memory: {reserved_memory / (1024 ** 3):.2f} GB")

    return f'cuda:{device}'



# ============ DATA UTILS ============
def load_syn_data(syn_data_path, data_loader=True, batch_size=256):
    checkpoint = torch.load(syn_data_path, weights_only=True)

    x_syn = checkpoint["x_syn"]
    y_syn = checkpoint["y_syn"]

    if data_loader:
        batch_size = int(min(256, batch_size))
        synthetic_dataset = torch.utils.data.TensorDataset(x_syn, y_syn)
        synthetic_loader = torch.utils.data.DataLoader(synthetic_dataset, 
                                                       batch_size=batch_size, 
                                                       shuffle=True)
        return synthetic_loader
    
    return x_syn, y_syn


### intialise synthetic data

class GetData(object):
    '''load random n samples from class c for time series data'''
    def __init__(self, tr_data, indices_class, device="cuda"):
        self.tr_data = tr_data
        self.indices_class = indices_class
        self.device = device

    def __call__(self, c, n):
        idx_shuffle = np.random.permutation(self.indices_class[c])[:n]

        return self.tr_data[idx_shuffle].to(self.device)  # Return n random samples for class c



def build_data_getter(tr_data, tr_lb, device, num_classes=2):
    '''build a real train data getter to load random n samples from class c'''

    num_classes = num_classes 
    indices_class = [[] for _ in range(num_classes)]  # Store indices per class

    for i, lab in enumerate(tr_lb):
        indices_class[int(lab)].append(i)

    tr_data = torch.FloatTensor(tr_data).to("cpu")
    data_getter = GetData(tr_data, indices_class, device=device)

    return data_getter



def get_data_array(dataloader):
    '''return dataset as numpy array from dataloader'''
    all_data = []
    all_labels =  []

    for batch, labels in dataloader:
        all_data.append(batch.numpy())  # Append batches (3D for time series)
        all_labels.extend(labels.numpy().tolist())  # Append labels

    data_array = np.concatenate(all_data, axis=0)  # Concatenate along batch dimension
    return data_array, all_labels


def initialize_synthetic_data(dataloader, ipc, device='cuda:0', initial_lr=0.01, num_classes=2,
                              dp_noise_scale=None, real=False, rand_real=False, seed=None):
    if seed: torch.manual_seed(seed)

    # Initialize with noise centred on real distribution (or real data)
    X_train = torch.cat([inputs.to(device) for inputs, _ in dataloader], dim=0)
    y_train = torch.cat([labels for _, labels in dataloader], dim=0)
    x_shape = X_train[0].shape   # (T, D) for time series or (D,) for flat

    x_syn_list = []
    y_syn_list = []

    for c in range(num_classes):
        X_c = X_train[y_train == c]
        if X_c.shape[0] == 0:
            raise ValueError(f"No samples found for class {c} in training data.")

        mu_c = X_c.mean(dim=0)
        std_c = X_c.std(dim=0) + 1e-5

        if dp_noise_scale is not None:
            mu_c += torch.normal(0, dp_noise_scale, size=mu_c.shape, device=device)
            std_c += torch.normal(0, dp_noise_scale, size=std_c.shape, device=device)

        if rand_real:
            # Sample synthetic from class normal distribution
            x_c_syn = mu_c + torch.randn((ipc, *x_shape), dtype=torch.float, device=device) * std_c
            
        else:
            # Sample from standard normal distribution
            x_c_syn = torch.randn((ipc, *x_shape), dtype=torch.float, device=device)

        y_c_syn = torch.full((ipc,), fill_value=c, dtype=torch.float, device=device)
        x_syn_list.append(x_c_syn)
        y_syn_list.append(y_c_syn)

    x_syn = torch.cat(x_syn_list, dim=0).requires_grad_(True)
    y_syn = torch.cat(y_syn_list, dim=0)

    # Initialize data with real samples
    if real:
        tr_data, tr_lb = get_data_array(dataloader)
        get_data = build_data_getter(tr_data, tr_lb, device)    

        for c in range(num_classes):
            real_x = get_data(c, None).detach().data  # get all real samples for class c
            n_real = real_x.shape[0]
            
            if n_real == 0:
                raise ValueError(f"No real samples found for class {c}.")
            
            else:
                sampled_x = real_x[torch.randperm(n_real)[:ipc]]

                # repeat some samples if there are not enough real samples to make up ipc
                if n_real<ipc: 
                    repeats = real_x[torch.randperm(ipc-n_real)]
                    sampled_x = torch.cat([sampled_x, repeats], dim=0) 
                    sampled_x = sampled_x[torch.randperm(ipc)]
            
            x_syn.data[c * ipc:(c + 1) * ipc] = sampled_x

    syn_lr = torch.tensor(float(initial_lr), device=device, dtype=torch.float32, requires_grad=True)

    return x_syn, y_syn, syn_lr


# ============ TRAINING AND EVALUATION ============
def prediction_binary(model, loader, loss_fn, device, apply_sigmoid=True):
    P = []
    L = []
    model.eval()
    val_loss = 0
    for i, batch in enumerate(loader):
        data, labels = batch
        data = data.to(torch.float32).to(device)
        labels = labels.to(torch.float32).to(device)
        
        pred = model(data)[:,0]
        if apply_sigmoid:
            pred = torch.sigmoid(pred)

        loss = loss_fn(pred,labels)
        val_loss = val_loss+loss.item()

        P.append(pred.cpu().detach().numpy())
        L.append(labels.cpu().detach().numpy())
        
    val_loss = val_loss/len(loader)                                 

    P = np.concatenate(P)  
    L = np.concatenate(L)
    auc = roc_auc_score(L, P)

    # apr = metrics.average_precision_score(L,P)
    precision, recall, _ = metrics.precision_recall_curve(L, P)
    apr = metrics.auc(recall, precision)

    return auc, val_loss, apr

def train_and_evaluate(net, image_syn, label_syn, val_loader, device, 
                       criterion, epochs=50, lr=0.05, syn_batch=256,
                       optim='sgd', mom=0.0, wd=5e-4, apply_sigmoid=False):
    
    # Initialize and optimizer
    if optim == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=mom, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    # Prepare synthetic data loader
    # Move evaluation copy to device
    image_syn_eval = image_syn.detach().clone().to(device)
    label_syn_eval = label_syn.detach().clone().to(device)

    syn_batch = int(min(256, syn_batch))
    synthetic_dataset = torch.utils.data.TensorDataset(image_syn_eval, label_syn_eval)
    synthetic_loader = torch.utils.data.DataLoader(
        synthetic_dataset, batch_size=syn_batch, shuffle=True)

    # Train on synthetic data
    net.train()
    for _ in range(epochs):
        running_loss = 0.0
        for inputs, targets in synthetic_loader:
            inputs, targets = inputs.to(device).to(torch.float32), targets.to(device).to(torch.float32)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = net(inputs)[:,0]

            if apply_sigmoid:
                outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss

    # Evaluate on the real test dataset
    net.eval()
    val_auc, val_loss, val_apr = prediction_binary(net, val_loader, criterion, device)

    return val_auc, val_loss, val_apr, net


def train_model(model, data_loader, criterion, optimizer, num_epochs=5,device='cuda'):
    model.train()
    for epoch in range(num_epochs):
        L=0
        for data, target in data_loader:
            data, target = data.to(torch.float32).to(device), target.to(torch.float32).to(device)
            optimizer.zero_grad()
            output = model(data)[:,0]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            L=L+loss.item()
        # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {L/len(data_loader):.4f}')
    return model,L/len(data_loader)