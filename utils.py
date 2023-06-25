from torch.utils.data import Dataset
import torch
from models import *

def select_model(model_name):
    if model_name=='deeponet':
        return deeponet
    elif model_name=='shiftdeeponet':
        return shiftdeeponet
    elif model_name=='flexdeeponet':
        return flexdeeponet
    elif model_name=='NOMAD':
        return NOMAD
    elif model_name=='hyperdeeponet':
        return hyperdeeponet

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        
class dataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        super(Dataset, self).__init__()

        self.x = x_tensor
        self.y = y_tensor
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
    
    
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def mse_loss(approx, real):
    return torch.mean((approx-real)**2)

def rel_loss(approx, real):
    return torch.mean(torch.abs(approx-real)/(torch.abs(real)+1e-8))
        
def rel_mse_error(approx, real):
    return torch.sum((approx-real)**2)/torch.sum((real)**2)

def rel_L2_error(approx, real):
    return torch.sum((approx-real)**2)**0.5/torch.sum((real)**2)**0.5