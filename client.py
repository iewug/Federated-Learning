import socket
from multiprocessing import Process
import argparse
import pickle
import dill
import os
from torch.utils.data import DataLoader
from network import Net
from utils import rcv_data, send_data

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int, help='which gpu')
parser.add_argument('--IP', default=socket.gethostbyname(socket.gethostname()), type=str, help='IP addr')
parser.add_argument('--PORT', default=12344, type=int, help='port number')
parser.add_argument('--N', default=3, type=int, help='client number')
parser.add_argument('--datadir', default='data/client', type=str, help='client data directory')
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
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

    # optimizer & criterion

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
        print(f"{name}: Receive weight from Server")
        weights = pickle.loads(weight_bytes)

        # train...
        lnet = Net().to(args.gpu) 
        lnet.load_state_dict(weights)
        lnet.train()
        # TODO

        # send local net weight to the server
        lweight = lnet.state_dict()
        lweightByte = pickle.dumps(lweight)
        send_data(lweightByte, client)
        print(f"{name}: Send weight to Server")


    print(f"{name}: Disconnected from Server")
    client.send("LOGOUT".encode())
    client.close()


processList = []
for i in range(args.N):
    p = Process(target=handle_server,args=(args,i+1))
    p.start()
    processList.append(p)

for process in processList:
    process.join()

