"""
source_timer: http://c.biancheng.net/view/2629.html 
source_scheduler: http://c.biancheng.net/view/2630.html
"""

import threading
import time
from threading import Timer

########### timer ############

def hello():
    print("hello, world")


# 指定10秒后执行hello函数
t = Timer(10.0, hello)
t.start()
count = 0


def print_time():
    print("当前时间：%s" % time.ctime())
    global t, count
    count += 1
    # 如果count小于10，开始下一次调度
    if count < 10:
        t = Timer(1, print_time)
        t.start()


# 指定1秒后执行print_time函数
t = Timer(1, print_time)
t.start()

########### scheduler ############
import threading
from sched import scheduler

def action(arg):
    print(arg)

#定义线程要调用的方法，*add可接收多个以非关键字方式传入的参数
def thread_action(*add):
    #创建任务调度对象
    sche = scheduler()
    #定义优先级
    i = 3
    for arc in add:
        # 指定1秒后执行action函数
        sche.enter(1, i, action,argument=(arc,))
        i = i - 1
    #执行所有调度的任务
    sche.run()

#定义为线程方法传入的参数
my_tuple = ("http://c.biancheng.net/python/",\
            "http://c.biancheng.net/shell/",\
            "http://c.biancheng.net/java/")
#创建线程
thread = threading.Thread(target = thread_action,args =my_tuple)
#启动线程
thread.start()



