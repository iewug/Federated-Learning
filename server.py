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
parser.add_argument('--PORT', default=12344, type=int, help='port number')
parser.add_argument('--M', default=3, type=int, help='update number')
parser.add_argument('--epoch', default=10, type=int, help='total epochs')
args = parser.parse_args()

# global var
glnet = Net().to(args.gpu)
updatecnt = 0
lnetWeightDict = {} #{addr:state_dict}
epoch = 0
threadcnt = 0
clientList = []
sendList = []

# condition
lock = threading.Lock()
cond = threading.Condition(lock)
sendcond = threading.Condition()

# thread
def handle_client(conn, addr):
    global updatecnt
    global epoch
    global threadcnt
    global sendList

    print(f"[NEW CONNECTION] {addr} connected")
    conn.send("Greeting from Server".encode('utf-8'))

    # sleep until all clients have been accepted
    with cond:
        threadcnt += 1
        clientList.append(addr)
        if threadcnt != args.N:
            cond.wait()
        else:
            print(f"{args.N} clients have been successfully accepted!")
            sendList = random.sample(clientList,args.M)
            print("*" * 50)
            print(f"Epoch{epoch}\nSend list: {sendList}")
            cond.notify_all()
            
    while True:
        with sendcond:
            while addr not in sendList:
                sendcond.wait()
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

        with cond:
            lnetWeightDict[addr] = weights
            updatecnt += 1
            if updatecnt != args.M:
                #print(f"{addr} waiting...")
                cond.wait()
            else:
                epoch += 1
                print("Evaluation")
                avgWeight = FedAvg(lnetWeightDict)
                glnet.load_state_dict(avgWeight)
                #eval...
                # TODO
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
            
        #print(f"{addr} waking...")

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
        # print(f"Active connections {threading.activeCount() - 1}")
except KeyboardInterrupt:
    print("\nServer exiting...")
finally:
    server.close()