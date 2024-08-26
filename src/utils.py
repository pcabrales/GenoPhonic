import random
import numpy as np
import torch

def set_seed(seed):
    """
    Set all the random seeds to a fixed value to take out any randomness
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    return True

def convert_to_list(value):
    if torch.is_tensor(value):
        return value.tolist()
    return value

