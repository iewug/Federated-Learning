'''
server.py
create N threads, one thread per client
partial participation mode:
    repeat
        - randomly choose M out of N clients
        - send global weight to M clients (where they train with local data)
        - receive local weight from them
        - average the weight from N clients (N-M use old weight)
type ctrl-c to quit
'''

import socket
import threading
import argparse
import pickle
from network import Net
import torch
import copy
import warnings
import os
import random
import torchvision
from utils import rcv_data, send_data

def FedAvg(weightDict):
    weightList = list(weightDict.values())
    w_avg = copy.deepcopy(weightList[0])
    for k in w_avg.keys():
        for i in range(1, len(weightList)):
            w_avg[k] += weightList[i][k]
        w_avg[k] = torch.div(w_avg[k], len(weightList))
    return w_avg

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int, help='which gpu')
parser.add_argument('--IP', default=socket.gethostbyname(socket.gethostname()), type=str, help='IP addr')
parser.add_argument('--PORT', default=12345, type=int, help='port number')
parser.add_argument('--M', default=10, type=int, help='update number')
parser.add_argument('--epoch', default=10, type=int, help='total epochs')
parser.add_argument('--batch-size', default=1000, type=int, help='test batch size')
args = parser.parse_args()

# global var
glnet = Net().to(args.gpu)
updatecnt = 0
lnetWeightDict = {} #{addr:state_dict}
epoch = 0
threadcnt = 0
clientList = []
sendList = []

# dataset
testset = torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ]))
testloader = torch.utils.data.DataLoader(testset,batch_size=args.batch_size, shuffle=True,num_workers=2,pin_memory=True)

# condition
# lock = threading.Lock()
cond = threading.Condition()
sendcond = threading.Condition()

# thread
def handle_client(conn, addr):
    global updatecnt
    global epoch
    global threadcnt
    global sendList

    print(f"[NEW CONNECTION] {addr} connected")
    conn.send("Greeting from Server".encode('utf-8'))

    # wait until all clients have been accepted
    with cond:
        threadcnt += 1
        clientList.append(addr)
        lnetWeightDict[addr] = glnet.state_dict()
        if threadcnt != args.N:
            cond.wait()
        else:
            print(f"{args.N} clients have been successfully accepted!")
            sendList = random.sample(clientList,args.M)
            print("*" * 50)
            print(f"Epoch{epoch}\nSend list: {sendList}")
            cond.notify_all()
            
    while True:
        # wait if not chosen
        with sendcond:
            while addr not in sendList:
                sendcond.wait()
        
        # send cmd
        if epoch == args.epoch:
            conn.send("LOGOUT".encode('utf-8'))
            break
        else:
            conn.send("CONTINUE".encode('utf-8'))
            conn.recv(1024)
    
        # send global net weight to the client
        glweight = glnet.state_dict()
        glweightByte = pickle.dumps(glweight)
        print(f"Send weight to {addr}")
        send_data(glweightByte, conn)

        # receive local net weight from the client
        weight_bytes = rcv_data(conn)
        print(f"Receive weight from {addr}")
        weights = pickle.loads(weight_bytes)

        # wait if receive weight's number < M
        # else do evaluation
        with cond:
            lnetWeightDict[addr] = weights
            updatecnt += 1
            if updatecnt != args.M:
                cond.wait()
            else:
                epoch += 1
                
                #evaluation
                print("Evaluation")
                avgWeight = FedAvg(lnetWeightDict)
                glnet.load_state_dict(avgWeight)
                glnet.eval()
                correct = 0
                with torch.no_grad():
                    for data, target in testloader:
                        data = data.to(args.gpu)
                        target = target.to(args.gpu)
                        output = glnet(data)
                        pred = output.max(1)[1]
                        correct += pred.eq(target).sum().item()
                print(f"ACC: {correct/len(testset)}")

                if epoch != args.epoch:
                    updatecnt = 0
                    sendList = random.sample(clientList,args.M)
                    print("*" * 50)
                    print(f"Epoch{epoch}\nSend list: {sendList}")
                else:
                    sendList = clientList
                with sendcond:
                    sendcond.notify_all()
                cond.notify_all()
            
    conn.recv(1024) # make sure client socket close first
    print(f"[DISCONNECTED] {addr} disconnected")
    conn.close()


# create listening socket
print("Server starting...")
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((args.IP,args.PORT))
server.listen()
print(f"Server is listening on {args.IP}:{args.PORT}.")


# get to know how many clients
try:
    conn, addr = server.accept()
    args.N = int(conn.recv(1024).decode('utf-8'))
    if args.N < args.M:
        warnings.warn(f"The number (M) of clients used for updating is greater than the total number (N) of clients ({args.M}>{args.N}). Here we set M=N")
        args.M = args.N
    conn.close()
except KeyboardInterrupt:
    print("\nServer exiting...")
    server.close()
    os._exit(0)


# a thread per client
try:
    print(f"Waiting to accept {args.N} clients...")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
except KeyboardInterrupt:
    print("\nServer exiting...")
finally:
    server.close()