import random
import torch
import numpy as np

def set_full_reproducibility(seed: int = 44) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def donor_robust_scale(adata, donor_key, layer="log1p", layer_out="log1p_donor_z",
                       trim=0.05, sd_floor_q=0.1, clip=5.0, scale=True):
    
    X = adata.layers[layer]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    Z = np.empty_like(X, dtype=np.float32)
    for d in adata.obs[donor_key].unique():
        idx = np.where(adata.obs[donor_key].values == d)[0]
        Xd = X[idx]
        mu = trim_mean(Xd, proportiontocut=trim, axis=0)       # robust centre, works for sparse genes
        if scale:
            sd = Xd.std(axis=0)
            floor = np.quantile(sd[sd > 0], sd_floor_q) if (sd > 0).any() else 1.0
            sd = np.maximum(sd, floor)                         # avoid exploding rare genes
        else:
            sd = 1.0
        Z[idx] = (Xd - mu) / sd
    adata.layers[layer_out] = np.clip(Z, -clip, clip)
    return adata