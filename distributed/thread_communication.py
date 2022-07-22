"""
source_condition: http://c.biancheng.net/view/vip_6191.html
source_event: http://c.biancheng.net/view/2626.html
"""

import threading
import time

############ condition #############

class Account:
    def __init__(self, account_no, balance) -> None:
        self.account_no = account_no
        self._balance = balance
        self.cond = threading.Condition()
        self._has_balance = False

    @property
    def balance(self):
        return self._balance

    # 提供一个线程安全的draw方法来完成取钱操作
    def draw(self, draw_amount):
        self.cond.acquire()
        try:
            if not self._has_balance or self._balance < draw_amount:
                # 导致当前线程进入 Condition 的等待池等待通知，并释放锁，直到其他线程调用该 Condition 的 notify() 或 notify_all() 方法来唤醒该线程。在调用该 wait() 方法时可传入一个 timeout 参数，指定该线程最多等待多少秒。
                self.cond.wait()
            else:
                print(f"{threading.current_thread().name} drew {draw_amount}.")
                self._balance -= draw_amount
                print(f"account balance:\t {self._balance}")
                self._has_balance = False
                self.cond.notify_all()
        finally:
            self.cond.release()

    def deposit(self, deposit_amount):
        self.cond.acquire()
        try:
            if self._has_balance:
                self.cond.wait()
            else:
                print(f"{threading.current_thread().name} deposited {deposit_amount}")
                self._balance += deposit_amount
                print(f"account balance:\t {self._balance}")
                self._has_balance = True
                self.cond.notify_all()
        finally:
            self.cond.release()


def draw_many(account: Account, draw_amount, max):
    for _ in range(max):
        account.draw(draw_amount)


def deposit_many(account: Account, deposit_amount, max):
    for _ in range(max):
        account.deposit(deposit_amount)


def test_condition():
    acct = Account("123", 0)
    thread_draw = threading.Thread(
        target=draw_many, name="drawer", args=(acct, 800, 100)
    )
    thread_draw.start()
    thread_depositor_1 = threading.Thread(
        target=deposit_many, name="depositor_1", args=(acct, 800, 100)
    )
    thread_depositor_1.start()
    thread_depositor_2 = threading.Thread(
        target=deposit_many, name="depositor_2", args=(acct, 800, 100)
    )
    thread_depositor_2.start()
    thread_depositor_3 = threading.Thread(
        target=deposit_many, name="depositor_3", args=(acct, 800, 100)
    )
    thread_depositor_3.start()


############ event #############

def test_event():
    event = threading.Event()
    def cal(name):
        # 等待事件，进入等待阻塞状态
        print('%s 启动' % threading.currentThread().getName())
        print('%s 准备开始计算状态' % name)
        event.wait()    # ①
        # 收到事件后进入运行状态
        print('%s 收到通知了.' % threading.currentThread().getName())
        print('%s 正式开始计算！'% name)
    # 创建并启动两条，它们都会①号代码处等待
    threading.Thread(target=cal, args=('甲', )).start()
    threading.Thread(target=cal, args=("乙", )).start()
    time.sleep(2)    #②
    print('------------------')
    # 发出事件
    print('主线程发出事件')
    event.set()


############ queue ###########
from queue import Queue
import random, threading, time

# 生产者类
class Producer(threading.Thread):
    def __init__(self, name, queue):
        threading.Thread.__init__(self, name=name)
        self.data = queue
    def run(self):
        for i in range(5):
            print("生产者 %s 将产品 %d 加入队列" % (self.getName(), i))
            self.data.put(i)
            time.sleep(random.random())
        print("生产者 %s 完成" % self.getName())

# 消费者类
class Consumer(threading.Thread):
    def __init__(self, name, queue):
        threading.Thread.__init__(self, name=name)
        self.data = queue
    def run(self):
        for i in range(5):
            val = self.data.get()
            print("消费者 %s 将产品 %d 从队列中取出" % (self.getName(), val))
            time.sleep(random.random())
        print("消费者 %s 完成" % self.getName())

if __name__ == '__main__':
    print("---主线程开始---")
    queue = Queue()                         # 实例化队列
    producer = Producer("Producer", queue)  # 实例化线程 Producer，并传入队列作为参数
    consumer = Consumer("Consumer", queue)  # 实例化线程 Consumer，并传入队列作为参数
    producer.start()                        # 启动线程 Producer
    consumer.start()                        # 启动线程 Consumer
    producer.join()                         # 等待线程 Producer 结束
    consumer.join()                         # 等待线程 Consumer 结束
    print("---主线程结束---")



# if __name__ == "__main__":
#     # test_condition()
#     # test_event()
#     pass
