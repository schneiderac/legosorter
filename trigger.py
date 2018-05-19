import time
import os
import shutil
import datetime
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

from imageclassifier import TFFP2

import socket

TCP_IP = '192.168.178.26'  # this IP of my pc. When I want raspberry pi 2`s as a client, I replace it with its IP '169.254.54.195'
TCP_PORT = 5008
BUFFER_SIZE = 1024

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))

def send_pos(position):
    MESSAGE = bytes(position, 'utf-8')
    s.send(MESSAGE)
    data = s.recv(BUFFER_SIZE)
    print("received data:", data)

class TestEventHandler(PatternMatchingEventHandler):
    classifier = TFFP2()

    def on_created(self, event):
        print (str(datetime.datetime.now()) + " " + str(event.src_path))

        file_name = event.src_path
        res = self.classifier.run(file_name)
        print('Result', res)

        part_name, confidence = res
        if confidence > 0.90:
            if part_name == 'verbinder':
                pos = '0'
            elif part_name == 'achsenende steck':
                pos = '2'
            else:
                pos = '1'
        else:
            pos = '1'

        send_pos(pos)
        print('\n\n')



if __name__ == '__main__':

    path = "/home/andreas/motion"
    event_handler = TestEventHandler(patterns=["*.jpg"])
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    observer.join()

    s.close()

