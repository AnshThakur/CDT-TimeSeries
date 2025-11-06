import torch
import torch.optim as optim
import numpy as np

def compute_loss(model, criterion, data_loader, params,device):
    state_dict = model.state_dict()
    new_state_dict = {k: v.clone() for k, v in zip(state_dict.keys(), params)}
    model.load_state_dict(new_state_dict)
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device).to(torch.float32), target.to(device).to(torch.float32)
            output = model(data)[:,0]
            loss = criterion(output, target)
            total_loss += loss.item()
            total_batches += 1
    return total_loss / total_batches


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

