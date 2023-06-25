'''
(완) 1. T = 30으로 변경 + interpolation으로 데이터 획득
2. phase 1 : 복원 loss + 상대 오차 추가
3. pahse2를 교수님이 새로 보내주신 논문 3.8식(가운데 w2, w3 부분 적분)을 참고 or solving pde 논문대로 해보기
4. 결과가 (교수님이 주신)논문처럼 t=3~4까지 감소하다가 거의 0이되면 성공?
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary

import numpy as np

import os
import os.path as osp
from tqdm import tqdm
import shutil 
import sys
import argparse
import pickle

from utils import *


def get_args(argv=None):
    parser = argparse.ArgumentParser(description = 'Put your hyperparameters')
    parser.add_argument('--name', default='',type=str, help='experiments name')
    parser.add_argument('--load_path', default='', type=str, help='path of directory to resume the training')

    parser.add_argument('--model', default=None, type=str, help='model name')    
    parser.add_argument('--data', default='identity', type=str, help='Data name')
    parser.add_argument('--gpu_idx', default=0, type=int, help='index of gpu which you want to use')
    parser.add_argument('--multgpu', action='store_true', help='whether multiple gpu or not')    
    #parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--seed', default=42,type=int, help='random seed')
    
    parser.add_argument('--batch', default=10000, type=int, help = 'batch size')
    parser.add_argument('--epochs', default=100000, type=int, help = 'Number of Epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--sche_type', default=None, type=str, help='type of scheduler: steplr, inversetime')  
    parser.add_argument('--step_size', default=100, type=int, help='scheduler step size')
    parser.add_argument('--gamma', default=0.99, type=float, help='scheduler factor')
    
    parser.add_argument('--d_in', default=1, type=int, help='dimension of input for target function')
    parser.add_argument('--d_out', default=1, type=int, help='dimension of output for target function')
    parser.add_argument('--n_sensor', default=100, type=int, help='number of sensor points')    
    parser.add_argument('--d_target', default=2, type=int, help='depth of target network (except basis)')
    parser.add_argument('--w_target', default=30, type=int, help='width of target network')
    parser.add_argument('--a_target', default='relu', type=str, help='activation of target network')
    parser.add_argument('--d_hyper', default=2, type=int, help='depth of hyper(branch) network (except basis)')
    parser.add_argument('--w_hyper', default=30, type=int, help='width of hyper(branch) network')
    parser.add_argument('--a_hyper', default='relu', type=str, help='activation of hyper(branch) network')
    parser.add_argument('--n_basis', default=100, type=int, help='number of basis (width of last layer in target network)')   
    
    parser.add_argument('--chunk_in', default=100, type=int, help='number of inputs for one chunk')  
    parser.add_argument('--chunk_out', default=100, type=int, help='number of outputs for one chunk')

    parser.add_argument('--custom',action="store_true")
    return parser.parse_args(argv)

if __name__=="__main__":
    args = get_args()
    #print(args)
    NAME = args.name
    CUSTOM = args.custom
    PATH = 'results/'
    #print(sys.argv) # input 체크

    if args.load_path is None:
        PATH = os.path.join(PATH, NAME)
        os.makedirs(PATH,exist_ok=True)
    else:
        #PATH = args.load_path
        args = torch.load(os.path.join(args.load_path, 'args.bin'))
        args.name = NAME
        PATH = os.path.join(PATH, NAME)
        os.makedirs(PATH,exist_ok=True)
    
    if CUSTOM:
        args.data = 'SIR_v2'
        args.multgpu = False
        args.gpu_idx = 0
        args.batch = 20000
        args.epochs = 20000
        #args.lr = 0.01
        args.n_sensor = 121
        args.d_out = 2
        args.d_in = 1
        loss_type = 'mse' # mse or rel
        lam = 30

    #shutil.copy(sys.argv[0], os.path.join(PATH, 'code.py'))
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.multgpu:
        num_gpu = torch.cuda.device_count()
    else:
        num_gpu = 1
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_idx)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ## setting
    batch_size = args.batch
    epochs = args.epochs
    lr = args.lr
    w_decay = args.wd
    scheduler_type = args.sche_type
    schedule_step_size = args.step_size
    schedule_gamma = args.gamma
    num_sensor = args.n_sensor
    #d_in = args.d_in
    #d_out = args.d_out
    #d_target = args.d_target
    #w_target = args.w_target
    #a_target = args.a_target
    #d_hyper = args.d_hyper
    #w_hyper = args.w_hyper
    #a_hyper = args.a_hyper
    #num_basis = args.n_basis

    torch.save(args, os.path.join(PATH, 'args.bin'))
    print(args)
    
    ## model
    model=select_model(args.model)(
                    args.d_target,
                    args.w_target,
                    args.a_target,
                    args.d_hyper,
                    args.w_hyper,
                    args.a_hyper,    
                    args.n_sensor,
                    args.d_in,
                    args.n_basis,
                    args.d_out
                ).cuda()
        
    model=model.cuda()
    print('The number of total parameters:', get_n_params(model))
    summary(model,[(1,),(num_sensor,)])
    
    ## load dataset
    if args.data=='shallow':
        N_train=100
    else:
        N_train=1000
    data_name=args.data+'_N'+str(N_train)+'_M'+str(num_sensor)+'.pickle'
    with open("./data_generation/"+data_name,"rb") as fr:
        raw_set= pickle.load(fr)
    train_dataset = TensorDataset(raw_set['train_X'].cuda().float(), raw_set['train_Y'].cuda().float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(raw_set['test_X'].cuda().float(), raw_set['test_Y'].cuda().float())
    test_loader = DataLoader(test_dataset, batch_size=raw_set['test_X'].shape[0], shuffle=False)

    if num_gpu> 1:
        print("Let's use", num_gpu, "GPUs!")
        model = nn.DataParallel(model).cuda()

    optimizer=torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
    
    if scheduler_type=='steplr':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=schedule_step_size, gamma=schedule_gamma)
    elif scheduler_type=='inversetime':
        fcn = lambda x: 1./(1. + schedule_gamma*x/schedule_step_size)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fcn)
    
    if loss_type == 'mse':
        loss_func = mse_loss
    elif loss_type == 'rel':
        loss_func = rel_loss

    os.makedirs(os.path.join(PATH, 'weights'),exist_ok=True)
    
    ## model trianing
    train_losses, train_losses_S, train_losses_I=[],[],[]
    test_losses=[]  
    test_rels_S, test_rels_I=[], []
    pbar = tqdm(total=epochs, file=sys.stdout)
    for epoch in tqdm(range(1,epochs+1)):
        ### train ###
        model.train()
        train_loss, train_loss_S, train_loss_I=AverageMeter(),AverageMeter(),AverageMeter()
        test_loss=AverageMeter()
        test_rel_S,test_rel_I=AverageMeter(), AverageMeter()
        for batch in train_loader:
            x,y=batch
            predict = model(x[:,num_sensor:],x[:,:num_sensor])

            S_loss=loss_func(predict[:,0],y[:,0])
            I_loss=loss_func(predict[:,1],y[:,1])
            loss = S_loss+lam*I_loss
            #zero gradients, backward pass, and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), y.shape[0])
            train_loss_S.update(S_loss.item(), y.shape[0])
            train_loss_I.update(I_loss.item(), y.shape[0])
        
        ### validation ###
        model.eval()
        with torch.no_grad():
            for batch in test_loader: # 모든 데이터 1개의 배치에, 1번만 반복됨
                x,y=batch
                predict=model(x[:,num_sensor:],x[:,:num_sensor])
                S_loss=loss_func(predict[:,0],y[:,0])
                I_loss=loss_func(predict[:,1],y[:,1])   
                error_test = S_loss+lam*I_loss
                test_loss.update(error_test.item(), y.shape[0])
                #print(predict.shape,y.shape)
                #print(predict.reshape(-1,args.n_sensor,2).shape,y.reshape(-1,args.n_sensor,2).shape)
                for prediction, real in zip(predict.reshape(-1,args.n_sensor,2),y.reshape(-1,args.n_sensor,2)):
                    test_rel_S.update(rel_L2_error(prediction[:,0],real[:,0]), 1) # 함수 1개에 대한 상대 오차 계산
                    test_rel_I.update(rel_L2_error(prediction[:,1],real[:,1]), 1)

        train_losses.append(train_loss.avg)
        train_losses_S.append(train_loss_S.avg)
        train_losses_I.append(train_loss_I.avg)

        test_losses.append(test_loss.avg)
        test_rels_S.append(test_rel_S.avg)
        test_rels_I.append(test_rel_I.avg)


        pbar.set_description("[Epoch : %d] Loss_train_S : %.5f, Loss_train_I : %.5f, Loss_train : %.5f, Loss_test : %.5f, rel_test_S : %.5f, rel_test_I : %.5f "%(epoch, train_losses_S[-1],train_losses_I[-1],train_losses[-1], test_losses[-1], test_rels_S[-1], test_rels_I[-1]))
        if scheduler_type!=None:
            scheduler.step()
        pbar.update()
        
        if epoch%1000==0:
            torch.save(model.state_dict(),os.path.join(PATH, 'weight_epoch_{}.bin'.format(epoch)))

            torch.save(model.state_dict(),os.path.join(PATH, 'weight.bin'))
            torch.save({'train_loss':train_losses,'train_loss_S':train_losses_S,'train_loss_I':train_losses_I, 'test_loss':test_losses, 'rel_test_S':test_rels_S, 'rel_test_I':test_rels_I}, os.path.join(PATH, 'loss.bin'))
            torch.save({'epoch':epoch,
                        'optimizer':optimizer.state_dict(),
                        'scheduler':scheduler.state_dict()}, os.path.join(PATH, 'checkpoint.bin'))
