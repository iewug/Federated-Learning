'''
client.py
create N processes, one process per client
- receive global weight
- train the local model on its local data
- send local weight to the server
'''

import socket
from multiprocessing import Process
import argparse
import pickle
import dill
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from network import Net
from utils import rcv_data, send_data

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int, help='which gpu')
parser.add_argument('--IP', default=socket.gethostbyname(socket.gethostname()), type=str, help='IP addr')
parser.add_argument('--PORT', default=12345, type=int, help='port number')
parser.add_argument('--N', default=20, type=int, help='client number')
parser.add_argument('--datadir', default='data/client', type=str, help='client data directory')
parser.add_argument('--batch-size', default=32, type=int, help='train batch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.5,type=float,help='SGD with momentum')
parser.add_argument('--epoch', default=5, type=int, help='total epochs')
args = parser.parse_args()

# notify the server of the client number
if args.N > 20:
    raise Exception(f"The number of clients shouldn't be over 20! Now we get n={args.N}.")
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((args.IP,args.PORT))
client.send(f"{args.N}".encode('utf-8'))
client.close()

# process
def handle_server(args,indx):
    # dataset
    datapath = os.path.join(args.datadir,f'Client{indx}.pkl')
    with open(datapath,'rb') as f:
        trainset = dill.load(f)
    trainloader = DataLoader(trainset,shuffle=True,batch_size=args.batch_size,num_workers=2,pin_memory=True)

    # local net
    lnet = Net().to(args.gpu)
    lnet.train()

    # optimizer & criterion
    optimizer = optim.SGD(lnet.parameters(), lr=args.lr,momentum=args.momentum)

    # socket
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((args.IP,args.PORT))
    name = client.getsockname()
    print(f"{name}: {client.recv(1024).decode('utf-8')}")
    

    while True:
        # receive command
        cmd = client.recv(1024).decode('utf-8')
        if cmd == 'LOGOUT':
            break
        else:
            client.send("CONTINUE".encode())

        # receive global net weight from the server
        weight_bytes = rcv_data(client)
        #print(f"{name}: Receive weight from Server")
        weights = pickle.loads(weight_bytes)

        # train...
        lnet.load_state_dict(weights)
        for epoch in range(args.epoch):
            totalloss = 0
            for data, target in trainloader:
                data = data.to(args.gpu)
                target = target.to(args.gpu)
                output = lnet(data)
                loss = F.nll_loss(output,target)
                totalloss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"{name}: Epoch{epoch}, loss{totalloss}")

        # send local net weight to the server
        lweight = lnet.state_dict()
        lweightByte = pickle.dumps(lweight)
        send_data(lweightByte, client)
        #print(f"{name}: Send weight to Server")

    # logout
    #print(f"{name}: Disconnected from Server")
    client.send("LOGOUT".encode())
    client.close()


# create process
processList = []
for i in range(args.N):
    p = Process(target=handle_server,args=(args,i+1))
    p.start()
    processList.append(p)
# wait all processes to end
for process in processList:
    process.join()

