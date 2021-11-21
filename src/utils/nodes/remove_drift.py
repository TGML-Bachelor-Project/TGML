import torch


def __try_numpy_to_tensor(t):
    return t if isinstance(t, torch.Tensor) else torch.from_numpy(t)


def center_z0(z0:torch.Tensor):
    z0 = __try_numpy_to_tensor(z0)
    return z0 - torch.mean(z0, dim=0)


def remove_v_drift(v0:torch.Tensor):
    v0 = __try_numpy_to_tensor(v0)
    return v0 - torch.mean(v0, dim=0)


def remove_rotation(z0:torch.Tensor, v0:torch.Tensor):
    z0 = __try_numpy_to_tensor(z0)
    v0 = __try_numpy_to_tensor(v0)
    zv = torch.vstack([z0, v0.reshape(-1,2)])
    U, S, VT = torch.linalg.svd(zv, full_matrices=False)
    new_coords = U * S.unsqueeze(0)
    ## z0 and v0 without rotation
    return new_coords[:z0.shape[0],:], new_coords[z0.shape[0]:,:].reshape(v0.shape)
