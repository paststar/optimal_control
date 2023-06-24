import torch.nn as nn
import torch

__all__ = ['flexdeeponet']

    
class flexdeeponet(nn.Module):
    def __init__(self, depth_trunk, width_trunk, act_trunk, depth_branch, width_branch, act_branch, num_sensor, input_dim, num_basis, output_dim=1):
        super(flexdeeponet, self).__init__()
        self.num_basis=num_basis
        self.input_dim=input_dim
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
        self.trunk_list.append(nn.Linear(input_dim,width_trunk))
        self.trunk_list.append(self.activation_trunk)
        for i in range(depth_trunk-1):
            self.trunk_list.append(nn.Linear(width_trunk, width_trunk))
            self.trunk_list.append(self.activation_trunk)
        self.trunk_list.append(nn.Linear(width_trunk,num_basis))
        self.trunk_list.append(self.activation_trunk)
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
        self.branch_list.append(nn.Linear(width_branch,(num_basis+1)*output_dim))
        self.branch_list = nn.Sequential(*self.branch_list)
        
        ##shift net
        self.shift_list = []
        self.shift_list.append(nn.Linear(num_sensor,width_branch))
        self.shift_list.append(self.activation_branch)
        for i in range(depth_branch-1):
            self.shift_list.append(nn.Linear(width_branch, width_branch))
            self.shift_list.append(self.activation_branch)
        self.shift_list.append(nn.Linear(width_branch,input_dim))
        self.shift_list = nn.Sequential(*self.shift_list)
        
        ##scale net
        self.scale_list = []
        self.scale_list.append(nn.Linear(num_sensor,width_branch))
        self.scale_list.append(self.activation_branch)
        for i in range(depth_branch-1):
            self.scale_list.append(nn.Linear(width_branch, width_branch))
            self.scale_list.append(self.activation_branch)
        self.scale_list.append(nn.Linear(width_branch,input_dim))
        self.scale_list = nn.Sequential(*self.scale_list)
        
        ##rotation net
        if input_dim==2:
            self.rotation_list = []
            self.rotation_list.append(nn.Linear(num_sensor,width_branch))
            self.rotation_list.append(self.activation_branch)
            for i in range(depth_branch-1):
                self.rotation_list.append(nn.Linear(width_branch, width_branch))
                self.rotation_list.append(self.activation_branch)
            self.rotation_list.append(nn.Linear(width_branch,1))
            self.rotation_list = nn.Sequential(*self.rotation_list)
        
    def forward(self, data_grid, data_sensor):
        branch_output=self.branch_list(data_sensor).reshape(-1,self.output_dim,self.num_basis+1)
        coeff=branch_output[...,:self.num_basis]
        bias=branch_output[...,-1]
        if self.input_dim==2:
            theta=self.rotation_list(data_sensor)
            #Need to fix
            rotation_matrix=torch.tensor([[torch.cos(theta),-torch.sin(theta)],[torch.sin(theta),torch.cos(theta)]])
            data_grid=torch.matmul(rotation_matrix.reshape(1,2,2), data_grid.reshape(-1,2,1)).squeeze(-1)
        if self.input_dim==3:
            #Need to fix
            theta=self.rotation_list(data_sensor)
            rotation_matrix=torch.tensor([[1,0,0],[0,torch.cos(theta),-torch.sin(theta)],[torch.sin(theta),torch.cos(theta)]])
            data_grid=torch.matmul(rotation_matrix.reshape(1,2,2), data_grid.reshape(-1,2,1)).squeeze(-1)
 
        scale_shift_rotation_grid=data_grid*self.scale_list(data_sensor)+self.shift_list(data_sensor)
        basis=self.trunk_list(scale_shift_rotation_grid).reshape(-1,1,self.num_basis).repeat(1,self.output_dim,1)
        
        y=torch.einsum("bij,bij->bi", coeff, basis)
        y=y+bias
        return y