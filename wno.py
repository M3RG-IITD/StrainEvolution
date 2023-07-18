## @meermehran  || M3RG Lab  || Indian Institute of Technology, Delhi
## @date : 18uly2023

##############################################################################
##                                                                          ##
## Revealing the Predictive Power of Neural Operators for Strain Evolution. ##
##                                                                          ##        
##############################################################################

##Import libraries
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from sklearn.metrics import r2_score

from Adam import Adam



torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



### WNO MODEL

class WNOConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, level, dummy):
        super(WNOConv2d_fast, self).__init__()

        """
        2D Wavelet layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.wave = 'db4'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.dwt_ = DWT(J=self.level, mode='symmetric', wave=self.wave).to(dummy.device)
        # print(dummy.shape)
        
        self.mode_data, _ = self.dwt_(dummy)
        # print(self.dwt_)
        self.modes1 = self.mode_data.shape[-2]
        self.modes2 = self.mode_data.shape[-1]
        # print(self.modes1, self.modes2)

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        
        # self.weights5 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    # Complex multiplication
    def mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        #Compute single tree Discrete Wavelet coefficients using some wavelet
        # dwt = DWT(J=3, mode='symmetric', wave='db6').cuda()
        dwt = DWT(J=self.level, mode='symmetric', wave=self.wave).to(device)

        x_ft, x_coeff = dwt(x)
        # print(x_ft.shape)

        # Multiply relevant Wavelet modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x_ft.shape[-2], x_ft.shape[-1], device=x.device)
        out_ft[:, :, :, :] = self.mul2d(x_ft[:, :, :, :], self.weights1)
        # Multiply the finer wavelet coefficients
        x_coeff[-1][:,:,0,:,:] = self.mul2d(x_coeff[-1][:,:,0,:,:].clone(), self.weights2)
        x_coeff[-1][:,:,1,:,:] = self.mul2d(x_coeff[-1][:,:,1,:,:].clone(), self.weights3)
        x_coeff[-1][:,:,2,:,:] = self.mul2d(x_coeff[-1][:,:,2,:,:].clone(), self.weights4)
        
        # x_coeff[-1][:,:,3,:,:] = self.mul2d(x_coeff[-1][:,:,2,:,:].clone(), self.weights5)
        
        # Return to physical space        
        idwt = IDWT(mode='symmetric', wave=self.wave).to(x.device)
        x = idwt((out_ft, x_coeff))
        
        return x

class WNO2d(nn.Module):
    def __init__(self, width, level, dummy_data):
        super(WNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Wavelet layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.level = level
        self.dummy_data = dummy_data
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3+2, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = WNOConv2d_fast(self.width, self.width, self.level, self.dummy_data)
        self.conv1 = WNOConv2d_fast(self.width, self.width, self.level, self.dummy_data)
        self.conv2 = WNOConv2d_fast(self.width, self.width, self.level, self.dummy_data)
        self.conv3 = WNOConv2d_fast(self.width, self.width, self.level, self.dummy_data)
        self.conv4 = WNOConv2d_fast(self.width, self.width, self.level, self.dummy_data)
        self.conv5 = WNOConv2d_fast(self.width, self.width, self.level, self.dummy_data)
        
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.w4 = nn.Conv2d(self.width, self.width, 1)
        self.w5 = nn.Conv2d(self.width, self.width, 1)
        
       

        self.fc1 = nn.Linear(self.width, 256)
        self.fc2 = nn.Linear(256, 1)
        
        self.act = nn.ReLU()

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)  ## [20,64,64,12]
        
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  ##[20,12, 64,64]
         

        

        x1 = self.conv0(x) ## ##[20,26, 64,64]
        x2 = self.w0(x)
        x = x1 + x2
        x = self.act(x)

        x1 = self.conv1(x)  ##[20,26, 64,64]
        
        x2 = self.w1(x)
        x = x1 + x2
        x = self.act(x)

        x1 = self.conv2(x) ##[20,26, 64,64]
        x2 = self.w2(x)
        x = x1 + x2
        x = self.act(x)

        x1 = self.conv3(x) #  [20,26, 64,64]]
        x2 = self.w3(x)
        x = x1 + x2
        x = self.act(x)

        x = x.permute(0, 2, 3, 1) #[20,64,64,26]
        x = self.fc1(x) ##[20,64,64,128]
        
        x = self.act(x)
        x = self.fc2(x)## [20,64,64,128]
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    


## Data loading
ntrain = 1000
ntest = 200

level = 1
width = 64

sub = 1
S = 48
T_in = 4
T = 10
step = 1

datafile = '<filepath>'

reader = MatReader(datafile)
data = reader.read_field('MatE22')
data = data.permute(0,2,3,1)

train_a = data[:ntrain,::sub,::sub,1:T_in]
train_u = data[:ntrain,::sub,::sub,T_in:T+T_in]

test_a = data[-ntest:,::sub,::sub,1:T_in]
test_u = data[-ntest:,::sub,::sub,T_in:T+T_in]

assert (S == train_u.shape[-2])
assert (T == train_u.shape[-1])

train_a = train_a.reshape(ntrain,S,S,T_in-1)
test_a = test_a.reshape(ntest,S,S,T_in-1)


print(f'train_a shape, train_u shape, :={train_a.shape}, {train_u.shape}')
print(f'test_a shape, test_u shape, :={test_a.shape}, {test_u.shape}')


batch_size = 20 # 10

epochs =1000
learning_rate = 0.0001
scheduler_step = 100
scheduler_gamma = 0.7

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), 
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), 
                                          batch_size=batch_size, shuffle=False)

## Model Definition
model = WNO2d(width, level, train_a.permute(0,3,1,2)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)


## Training and Testing
trainloss=[]
timetaken=0
testloss =[]

myloss = LpLoss(size_average=False)
for ep in range(1, epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
#            

            if t == 0:
#                 pred = im
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
#             

            xx = torch.cat((xx[..., step:], y), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        
        train_l2_full += l2_full.item()
    

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    trainloss.append(train_l2_full/ntrain)
    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            
            loss = 0
            xx = xx.to(device)
#           
            yy = yy.to(device)

            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
#                 
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)
#                 print(step)
                xx = torch.cat((xx[..., step:], im), dim=-1)
#                 print('2ndone')
#                 print(xx.shape)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
    testloss.append(test_l2_full/ntest)

    t2 = default_timer()
    timetaken+=(t2-t1)
    scheduler.step()

    print(ep,t2-t1,train_l2_full/ntrain, test_l2_full/ntest)
    
print(f'total time taken=={timetaken/60}')