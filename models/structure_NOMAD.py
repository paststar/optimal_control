import torch.nn as nn
import torch

__all__ = ['NOMAD']


    
class NOMAD(nn.Module):
    def __init__(self, depth_trunk, width_trunk, act_trunk, depth_branch, width_branch, act_branch, num_sensor, input_dim, num_basis, output_dim):
        super(NOMAD, self).__init__()
        self.num_basis=num_basis
        self.output_dim=output_dim
        
        ##trunk net
        if act_trunk=='tanh':
            self.activation_trunk=nn.Tanh()
        elif act_trunk=='prelu':
            self.activation_trunk=nn.PReLU()
        elif act_trunk=='relu':
            self.activation_trunk=nn.ReLU()
        else:
            print('error!!')
        self.trunk_list = []
        self.trunk_list.append(nn.Linear(input_dim+num_basis,width_trunk))
        self.trunk_list.append(self.activation_trunk)
        for i in range(depth_trunk-1):
            self.trunk_list.append(nn.Linear(width_trunk, width_trunk))
            self.trunk_list.append(self.activation_trunk)
        self.trunk_list.append(nn.Linear(width_trunk,num_basis))
        self.trunk_list.append(self.activation_trunk)
        self.trunk_list.append(nn.Linear(num_basis,output_dim))
        self.trunk_list = nn.Sequential(*self.trunk_list)
        
        ##branch net
        if act_branch=='tanh':
            self.activation_branch=nn.Tanh()
        elif act_branch=='prelu':
            self.activation_branch=nn.PReLU()
        elif act_branch=='relu':
            self.activation_branch=nn.ReLU()
        else:
            print('error!!')
        self.branch_list = []
        self.branch_list.append(nn.Linear(num_sensor,width_branch))
        self.branch_list.append(self.activation_branch)
        for i in range(depth_branch-1):
            self.branch_list.append(nn.Linear(width_branch, width_branch))
            self.branch_list.append(self.activation_branch)
        self.branch_list.append(nn.Linear(width_branch,num_basis))
        self.branch_list = nn.Sequential(*self.branch_list)
        

        
    def forward(self, data_grid, data_sensor):
        coeff=self.branch_list(data_sensor).reshape(-1,self.num_basis)
        output=self.trunk_list(torch.cat([coeff,data_grid],dim=-1))
        
        return output