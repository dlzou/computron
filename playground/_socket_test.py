from dataclasses import dataclass
import socket
import sys


def client(port):
    s = socket.socket()
    s.connect(('127.0.0.1', port))
    while True:
        str = input("S: ")
        s.send(str.encode());
        if str == "bye":
            break
        print("N:", s.recv(1024).decode())
    s.close()


def server(port):
    s = socket.socket()
    s.bind(('', port))
    s.listen(5)
    c, addr = s.accept()
    print("Socket up and running with a connection from", addr)
    while True:
        rcvdData = c.recv(1024).decode()
        print("S: ", rcvdData)
        sendData = input("N: ")
        c.send(sendData.encode())
        if sendData == "bye":
            break
    c.close()


if __name__ == "__main__":
    port = 29600
    if sys.argv[1] == "client":
        client(port)
    elif sys.argv[1] == "server":
        server(port)
