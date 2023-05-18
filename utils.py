def send_data(data, r):
    length = len(data)
    r.send(str(length).encode("utf-8"))
    r.recv(1024) # 避免粘包
    r.sendall(data)


def rcv_data(sender):
    length = int(sender.recv(1024).decode("utf-8"))
    sender.send("Received".encode("utf-8"))
    ret = b''
    while len(ret) < length:
        weight_byte = sender.recv(1024)
        ret += weight_byte
    return ret
