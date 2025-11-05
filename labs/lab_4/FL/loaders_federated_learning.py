import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle as cPickle

def get_loaders(
    path='../DATA',
    nodes=5, 
    batch=64,
    min_samples_per_class=10,
    alpha=0.5
):
    """
    Create non-IID DataLoaders for each client (node) for federated learning simulation
    using the In-Hospital Mortality (IHM) dataset.

    Parameters
    ----------
    path : str
        Path to the directory containing train_raw.p, val_raw.p, and test_raw.p
    nodes : int
        Number of simulated clients (federated learning nodes)
    batch : int
        Batch size for the DataLoaders
    min_samples_per_class : int
        Minimum number of samples per class to allocate to each node
    alpha : float
        Dirichlet distribution concentration parameter controlling non-IIDness.
        Lower alpha → more non-IID (data skewed towards some classes per node)
        Higher alpha → more IID (data distributed more evenly)

    Returns
    -------
    Loaders : list
        List of [train_loader, val_loader, test_loader] for each node
    C : np.ndarray
        Array of client data proportions (used for weighting in FL aggregation)
    """

    def split_data_non_iid(data, labels, nodes, min_samples_per_class, alpha):
        """
        Split data into non-IID partitions for federated learning using Dirichlet sampling.

        Each client gets:
        - At least `min_samples_per_class` samples per class (if available)
        - Remaining samples distributed according to Dirichlet(alpha)
        """

        data = np.array(data)
        labels = np.array(labels)

        # Get indices for each class (assuming binary classification)
        label_indices = [np.where(labels == i)[0] for i in range(2)]

        # Initialize client-specific index lists
        client_data_indices = [[] for _ in range(nodes)]

        # Step 1: Assign a minimum number of samples per class to each client
        for class_id, indices in enumerate(label_indices):
            np.random.shuffle(indices)
            for node_idx in range(nodes):
                allocated = min(min_samples_per_class, len(indices))
                client_data_indices[node_idx].extend(indices[:allocated].tolist())
                indices = indices[allocated:]

        # Step 2: Distribute remaining samples using Dirichlet(alpha)
        for class_id, indices in enumerate(label_indices):
            if len(indices) == 0:
                continue
            # Generate class proportions across clients
            portions = np.random.dirichlet(np.ones(nodes) * alpha)
            # Split remaining indices accordingly
            split_indices = np.split(
                indices, (np.cumsum(portions)[:-1] * len(indices)).astype(int)
            )
            for node_idx, node_split in enumerate(split_indices):
                client_data_indices[node_idx].extend(node_split.tolist())

        # Step 3: Gather the actual data and labels for each client
        node_data = [data[idx] for idx in client_data_indices]
        node_labels = [labels[idx] for idx in client_data_indices]

        return node_data, node_labels

    # -----------------------------
    # Load Training Data
    # -----------------------------
    with open(f'{path}/train_raw.p', 'rb') as f:
        x = cPickle.load(f)
    train_data, train_labels = x[0], x[1]
    Train_T, Train_L = split_data_non_iid(train_data, train_labels, nodes, min_samples_per_class, alpha)

    # -----------------------------
    # Load Validation Data
    # -----------------------------
    with open(f'{path}/val_raw.p', 'rb') as f:
        x = cPickle.load(f)
    val_data, val_labels = x[0], x[1]
    Val_T, Val_L = split_data_non_iid(val_data, val_labels, nodes, min_samples_per_class, alpha)

    # -----------------------------
    # Load Test Data
    # -----------------------------
    with open(f'{path}/test_raw.p', 'rb') as f:
        x = cPickle.load(f)
    test_data, test_labels = x["data"][0], np.array(x["data"][1])
    Test_T, Test_L = split_data_non_iid(test_data, test_labels, nodes, min_samples_per_class, alpha)

    # -----------------------------
    # Build DataLoaders for each node
    # -----------------------------
    Loaders = []
    Count = []

    for i in range(nodes):
        num_samples = 0

        # Training data
        train_dataset = TensorDataset(torch.tensor(Train_T[i]), torch.tensor(Train_L[i]))
        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
        num_samples += len(Train_L[i])

        # Validation data
        val_dataset = TensorDataset(torch.tensor(Val_T[i]), torch.tensor(Val_L[i]))
        val_loader = DataLoader(val_dataset, batch_size=batch)
        num_samples += len(Val_L[i])

        # Test data
        test_dataset = TensorDataset(torch.tensor(Test_T[i]), torch.tensor(Test_L[i]))
        test_loader = DataLoader(test_dataset, batch_size=batch)
        num_samples += len(Test_L[i])

        Loaders.append([train_loader, val_loader, test_loader])
        Count.append(len(Train_L[i]))

        # Compute positive class ratio (fraction of label=1 samples)
        all_labels = np.concatenate([Train_L[i], Val_L[i], Test_L[i]])
        positive_ratio = np.mean(all_labels)

        # Informative client summary
        print(f'Client {i+1}/{nodes} Summary:')
        print(f'  Total samples: {num_samples}')
        print(f'  Training samples: {len(Train_L[i])}, Validation: {len(Val_L[i])}, Test: {len(Test_L[i])}')
        print(f'  Positive class ratio: {positive_ratio:.3f} ({positive_ratio*100:.1f}%)')
        print('-----------------------------------')


    # Proportion of training data per client
    C = np.array(Count) / sum(Count)

    return Loaders, C
