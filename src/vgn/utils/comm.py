import time
from multiprocessing.connection import Listener, Client

def receive_msg(port=12345):
    address = ('localhost', port)     # family is deduced to be 'AF_INET'
    listener = Listener(address, authkey=b'secret password')
    print('Waiting to connect...')
    conn = listener.accept()
    print('connection accepted from', listener.last_accepted)
    msgs = []
    while True:
        msg = conn.recv()
        # do something with msg

        print(msg)
        if msg == 'close':
            conn.close()
            break
        msgs.append(msg)
    listener.close()
    return msgs

def send_msg(msgs, port=12346):
    address = ('localhost', port)
    while True:
        try:
            conn = Client(address, authkey=b'secret password')
            break
        except ConnectionRefusedError:
            print('Connection refused, retry in 3s')
            time.sleep(3)
            pass
    for msg in msgs:
        conn.send(msg)
    conn.send('close')
    conn.close()
