import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

__all__ = ['hyperdeeponet', 'chunk_hyperdeeponet']


class FC_layer(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        
        self.num_in = num_in
        self.num_out = num_out
        self.weight_size=torch.tensor([num_out,num_in])
        self.bias_size=torch.tensor([num_out])
        
    def forward(self, x, param):
        B=x.shape[0]
        w=param[:,:torch.prod(self.weight_size)]
        b=param[:,torch.prod(self.weight_size):]
        
        w=w.reshape(B,self.weight_size[0],self.weight_size[1])
        b=b.reshape(B,self.bias_size)
        return torch.einsum('ab,acb->ac', x, w) + b
    
    def get_weight_size(self):
        return self.weight_size

    def get_bias_size(self):
        return self.bias_size
    
    def get_param_size(self):
        return torch.prod(self.weight_size)+self.bias_size


class hyperdeeponet(nn.Module):
    def __init__(self, depth_trunk, width_trunk, act_trunk, depth_hyper, width_hyper, act_hyper, num_sensor, input_dim, num_basis, output_dim, constraint=False):
        super(hyperdeeponet, self).__init__()
        
        ##trunk net = target net
        self.depth_trunk=depth_trunk
        if act_trunk=='tanh':
            self.activation_trunk=nn.Tanh()
        elif act_trunk=='prelu':
            self.activation_trunk=nn.PReLU()
        elif act_trunk=='relu':
            self.activation_trunk=nn.ReLU()
        else:
            print('error!!')
        self.trunk_list = []
        self.param_sizes=[]
        self.trunk_list.append(FC_layer(input_dim,width_trunk))
        self.param_sizes.append(FC_layer(input_dim,width_trunk).get_param_size())
        for i in range(depth_trunk-1):
            self.trunk_list.append(FC_layer(width_trunk, width_trunk))
            self.param_sizes.append(FC_layer(width_trunk, width_trunk).get_param_size())
        self.trunk_list.append(FC_layer(width_trunk,num_basis))
        self.param_sizes.append(FC_layer(width_trunk,num_basis).get_param_size())
        self.trunk_list.append(FC_layer(num_basis,output_dim))
        self.param_sizes.append(FC_layer(num_basis,output_dim).get_param_size())
        self.param_size=int(sum(self.param_sizes))
        
        ##hyper net
        self.depth_hyper=depth_hyper
        if act_hyper=='tanh':
            self.activation_hyper=nn.Tanh()
        elif act_hyper=='prelu':
            self.activation_hyper=nn.PReLU()
        elif act_hyper=='relu':
            self.activation_hyper=nn.ReLU()
        else:
            print('error!!')
        self.hyper_list = []
        self.hyper_list.append(nn.Linear(num_sensor,width_hyper))
        self.hyper_list.append(self.activation_hyper)
        for i in range(depth_hyper-1):
            self.hyper_list.append(nn.Linear(width_hyper, width_hyper))
            self.hyper_list.append(self.activation_hyper)
        self.hyper_list.append(nn.Linear(width_hyper,self.param_size))
        self.hyper_list = nn.Sequential(*self.hyper_list)

        print('constraint :',constraint)

        self.constraint = constraint
        
    def forward(self, data_grid, data_sensor):
        cut=0
        weight=self.get_param(data_sensor)
        for i in range(self.depth_trunk+1):
            data_grid=self.trunk_list[i](data_grid, weight[...,cut:cut+self.param_sizes[i]])
            data_grid=self.activation_trunk(data_grid)
            cut+=self.param_sizes[i]
        
        output=self.trunk_list[self.depth_trunk+1](data_grid, weight[...,cut:cut+self.param_sizes[self.depth_trunk+1]])

        if self.constraint:
            output = F.relu(output)
        return output

    
    def get_param(self, data):
        return self.hyper_list(data)

    
class chunk_hyperdeeponet(nn.Module):
    def __init__(self, depth_trunk, width_trunk, act_trunk, depth_hyper, width_hyper, act_hyper, num_sensor, input_dim, num_basis, output_dim, num_chunk_in, num_chunk_out):
        super(chunk_hyperdeeponet, self).__init__()
        
        ##trunk net = target net
        self.depth_trunk=depth_trunk
        if act_trunk=='tanh':
            self.activation_trunk=nn.Tanh()
        elif act_trunk=='prelu':
            self.activation_trunk=nn.PReLU()
        elif act_trunk=='relu':
            self.activation_trunk=nn.ReLU()
        else:
            print('error!!')
        self.trunk_list = []
        self.param_sizes=[]
        self.trunk_list.append(FC_layer(input_dim,width_trunk))
        self.param_sizes.append(FC_layer(input_dim,width_trunk).get_param_size())
        for i in range(depth_trunk-1):
            self.trunk_list.append(FC_layer(width_trunk, width_trunk))
            self.param_sizes.append(FC_layer(width_trunk, width_trunk).get_param_size())
        self.trunk_list.append(FC_layer(width_trunk,num_basis))
        self.param_sizes.append(FC_layer(width_trunk,num_basis).get_param_size())
        self.trunk_list.append(FC_layer(num_basis,output_dim))
        self.param_sizes.append(FC_layer(num_basis,output_dim).get_param_size())
        self.param_size=int(sum(self.param_sizes))
        
        ##chunk
        self.num_sensor=num_sensor
        self.num_chunk_in=num_chunk_in
        self.num_chunk_out=num_chunk_out
        self.num_chunk=int(np.ceil(self.param_size/num_chunk_out))
        self.latent_chunk=torch.nn.Parameter(torch.randn(self.num_chunk, num_chunk_in))
        
        ##hyper net
        self.depth_hyper=depth_hyper
        if act_hyper=='tanh':
            self.activation_hyper=nn.Tanh()
        elif act_hyper=='prelu':
            self.activation_hyper=nn.PReLU()
        elif act_hyper=='relu':
            self.activation_hyper=nn.ReLU()
        else:
            print('error!!')
        self.hyper_list = []
        self.hyper_list.append(nn.Linear(num_sensor+num_chunk_in,width_hyper))
        self.hyper_list.append(self.activation_hyper)
        for i in range(depth_hyper-1):
            self.hyper_list.append(nn.Linear(width_hyper, width_hyper))
            self.hyper_list.append(self.activation_hyper)
        self.hyper_list.append(nn.Linear(width_hyper,num_chunk_out))
        self.hyper_list = nn.Sequential(*self.hyper_list)
        
    def forward(self, data_grid, data_sensor):
        B=data_sensor.shape[0]
        weight=self.get_param(torch.cat((data_sensor.reshape(B,1,self.num_sensor).repeat(1,self.num_chunk,1).reshape(-1,self.num_sensor),self.latent_chunk.repeat(B,1)),dim=-1))

        weight=weight.reshape(B,-1)
        cut=0
        for i in range(self.depth_trunk+1):
            data_grid=self.trunk_list[i](data_grid, weight[...,cut:cut+self.param_sizes[i]])
            data_grid=self.activation_trunk(data_grid)
            cut+=self.param_sizes[i]
        
        output=self.trunk_list[self.depth_trunk+1](data_grid, weight[...,cut:cut+self.param_sizes[self.depth_trunk+1]])
        return output

    
    def get_param(self, data):
        return self.hyper_list(data)
    
    
