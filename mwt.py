## @meermehran  || M3RG Lab  || Indian Institute of Technology, Delhi
## @date : 18uly2023

##############################################################################
##                                                                          ##
## Revealing the Predictive Power of Neural Operators for Strain Evolution. ##
##                                                                          ##        
##############################################################################

##Import libraries
from mwtmodel import *
from mwtutils import *

torch.manual_seed(0)
np.random.seed(0)

ntrain = 1000
ntest = 200

level = 2
# width = 32
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

sub = 1
S = 48
T_in = 4
T = 10
step = 1



path = '<filepath>'

data = sio.loadmat(path)['MatE22']

data = data.transpose(0,2,3,1).astype(np.float32)
data = torch.from_numpy(data)



train_a = data[:ntrain,::sub,::sub,1:T_in]
train_u = data[:ntrain,::sub,::sub,T_in:T+T_in]

test_a = data[-ntest:,::sub,::sub,1:T_in]
test_u = data[-ntest:,::sub,::sub,T_in:T+T_in]

# print(train_u.shape)
# print(test_u.shape)
assert (S == train_u.shape[-2])
assert (T == train_u.shape[-1])

train_a = train_a.reshape(ntrain,S,S,T_in-1)
test_a = test_a.reshape(ntest,S,S,T_in-1)


## Model Initialization



batch_size = 40 # 10


epochs = 100
learning_rate = 0.001
scheduler_step = 50
scheduler_gamma = 0.5

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

ich = 3
initializer = get_initializer('xavier_normal') # xavier_normal, kaiming_normal, kaiming_uniform
model = MWT2d(ich,
            alpha = 10,#10, 
            c =4,#4,#4*4,
            k = 4,
            base = 'legendre', # chebyshev# legendre
            nCZ = 2,#2,
            L=0,
            initializer = initializer,
            ).to(device) 


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)


myloss = LpLoss(size_average=False) 



epochs =500
trainloss =[]
testloss = []
timetaken=0
pad = 16
for epoch in range(1, epochs+1):
    model.train()
    # total_loss = 0
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        xx, yy = xx.to(device), yy.to(device)
        loss = 0
        for t in range(0, T, step):
            y = yy[..., t:t+step]

            xpad = F.pad(xx, [0,0,0,pad,pad,0])  #bs,64,64,1
            ypad = F.pad(y, [0,0,0,pad,pad,0])
           
            output = model(xpad).unsqueeze(-1)#[:,pad:,:-pad,:]   # 20,64,64,1
            # print(output.shape, ypad.shape, y.shape, xpad.shape)
            
            

            loss += myloss(output.reshape(batch_size,-1), ypad.reshape(batch_size,-1))
            if t==0:
                pred =output[:,pad:,:-pad,:]
                # print(pred[:,pad:,:-pad,:].shape, y.shape)
                

            else:
                pred = torch.cat((pred,output[:,pad:,:-pad,:]) , -1)
                
            xx = torch.cat((xx[...,step:], y), dim =-1)   
            
                        
#             
        train_l2_step += loss.item()
        # print(pred.shape, yy.shape)
        l2_full = myloss(pred.reshape(batch_size,-1), yy.reshape(batch_size,-1)).item()
        train_l2_full += l2_full
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    trainloss.append(train_l2_full/ntrain)    
    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        fulltestloss = 0
        model.eval()
        all_steps = torch.zeros(ntest, S, S,T)
        c = 0
        for xx, yy in test_loader:
            

            xx, yy = xx.to(device), yy.to(device)
            loss = 0
            fulltestloss = 0
            for t in range(0, T, step):
                y = yy[..., t:t+step]

                xpad = F.pad(xx, [0,0,0,pad,pad,0])
                ypad = F.pad(y, [0,0,0,pad,pad,0])   #[20,64,64,1]

                output = model(xpad).unsqueeze(-1)#[:,pad:,:-pad,:]   # 20,64,64,1

                loss += myloss(output.reshape(batch_size,-1), ypad.reshape(batch_size,-1))
                if t==0:
                    pred = output[:,pad:,:-pad,:]
                  

                else:
                    pred = torch.cat((pred,output[:,pad:,:-pad,:]) , -1)#[:,pad:,:-pad,:]
                xx = torch.cat((xx[...,step:], output[:,pad:,:-pad,:] ), dim =-1)  

            
            all_steps[ batch_size *c : batch_size*c +batch_size] = pred
            c += 1
        
            t_l2_full = myloss(pred.reshape(batch_size,-1), yy.reshape(batch_size,-1)).item()
            test_l2_step += loss.item()

            
            test_l2_full += t_l2_full
        testloss.append(test_l2_full/ntest)
    fulltestloss = myloss(all_steps.reshape(ntest,-1), test_u.reshape(ntest,-1)).item()
    t2 = default_timer()
    
    # torch.save(model, f'nTrain/n300/MWT_T10_ep_{epoch}.pt')
    
    
    timetaken+=(t2-t1)
    scheduler.step()   
    print(epoch,t2-t1,train_l2_full/ntrain, test_l2_full/ntest, fulltestloss/ntest)
print(f'total time taken=={timetaken/60}')