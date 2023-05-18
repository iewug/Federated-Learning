import os
import argparse
import socket
import dill
from network import Net
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int, help='which gpu')
parser.add_argument('--IP', default=socket.gethostbyname(socket.gethostname()), type=str, help='IP addr')
parser.add_argument('--PORT', default=12344, type=int, help='port number')
parser.add_argument('--N', default=3, type=int, help='client number')
parser.add_argument('--datadir', default='data/client', type=str, help='client data directory')
parser.add_argument('--momentum', default=0.5,type=float,help='SGD with momentum')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
args = parser.parse_args()

indx = 1
datapath = os.path.join(args.datadir,f'Client{indx}.pkl')
print(datapath)
with open(datapath,'rb') as f:
    trainset = dill.load(f)
trainloader = DataLoader(trainset,shuffle=True,batch_size=args.batch_size,num_workers=2,pin_memory=True)

net = Net().to(args.gpu)
optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=args.momentum)

for epoch in range(5):
    print(epoch)
    net.train()
    for data, target in trainloader:
        data = data.to(args.gpu)
        target = target.to(args.gpu)
        output = net(data)
        loss = F.nll_loss(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
    net.eval()
    correct = 0
    with torch.no_grad():
        for data, target in trainloader:
            data = data.to(args.gpu)
            target = target.to(args.gpu)
            output = net(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    print("ACC",epoch)
    print(correct/len(trainset))
        
