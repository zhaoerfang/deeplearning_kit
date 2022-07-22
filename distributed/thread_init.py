import threading
from typing import *


def action(*add):
    for arc in add:
        print(threading.current_thread().getName() + " " + arc)


my_tuple = ("url_1", "url_2", "url_3")


def run_thread_1():
    thread = threading.Thread(target=action, args=my_tuple)
    thread.start()
    # join() 方法的功能是在程序指定位置，优先让该方法的调用者使用 CPU 资源， timeout是线程最多可以霸占cpu资源的时间，省略则是默认直到线程执行结束才释放cpu资源
    thread.join() 
    for i in range(5):
        print(threading.current_thread().getName())


class my_Thread(threading.Thread):
    def __init__(self, add):
        super().__init__()
        self.add = add

    def run(self):
        for arc in self.add:
            print(threading.current_thread().getName() + " " + arc)


def run_thread_2():
    my_thread = my_Thread(my_tuple)
    my_thread.start()
    for i in range(5):
        print(threading.current_thread().getName())


run_thread_2()
