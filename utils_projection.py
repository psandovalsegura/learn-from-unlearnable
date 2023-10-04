import numpy as np
import torch

def get_projection_vectors(vectors):
    normalized_vectors = []
    for vec in vectors:
        norm = np.linalg.norm(vec)
        normalized_vectors.append(vec/norm)
    projection_vectors = np.stack(normalized_vectors)
    return projection_vectors

def get_q_matrix(vectors):
    projection_vectors = get_projection_vectors(vectors)
    q, r = np.linalg.qr(projection_vectors.T)
    return q.T

def project_data(data, vectors, use_q_matrix=False):
    projected_data = data.copy()
    if use_q_matrix:
        projection_vectors = get_q_matrix(vectors)
    else:
        projection_vectors = get_projection_vectors(vectors)
    for projection_vec in projection_vectors:
        dot_product = projected_data @ projection_vec
        projected_data -= (dot_product).reshape((-1, 1)) * projection_vec.reshape((1, -1))
    return projected_data

def subtract_margin(data, vectors):
    projected_data = data.copy()
    projection_vectors = get_projection_vectors(vectors)
    dot_products = []
    for projection_vec in projection_vectors:
        dot_product = projected_data @ projection_vec
        dot_products.append(dot_product)
    dot_products = np.stack(dot_products)
    mean_dot_products = np.mean(dot_products, axis=0)
    
    return projected_data

class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform:
            x = self.transform(x)
        y = self.tensors[1][index]
        return x, y
    def __len__(self):
        return self.tensors[0].size(0)