# source: http://c.biancheng.net/view/2617.html

"""
多线程编程是一件有趣的事情，它很容易突然出现“错误情况”，这是由系统的线程调度具有一定的随机性造成的。不过，即使程序偶然出现问题，那也是由于编程不当引起的。当使用多个线程来访问同一个数据时，很容易“偶然”出现线程安全问题。


当两个线程相互等待对方释放资源时，就会发生死锁。Python 解释器没有监测，也不会主动采取措施来处理死锁情况，所以在进行多线程编程时应该采取措施避免出现死锁。
一旦出现死锁，整个程序既不会发生任何异常，也不会给出任何提示，只是所有线程都处于阻塞状态，无法继续。
"""

import threading
import time


class Account:
    def __init__(self, account_no, balance) -> None:
        self.account_no = account_no
        self.balance = balance


def draw(account: Account, draw_amount):
    if account.balance >= draw_amount:
        print(f"{threading.current_thread().name}取钱成功：\t{draw_amount}")
        # time.sleep(0.001)
        account.balance -= draw_amount
        print(f"余额为:\t {account.balance}")
    else:
        print(f"{threading.current_thread().name}取钱失败，余额不足。")


acct = Account("1234567", 1000)

thread_1 = threading.Thread(name="甲", target=draw, args=(acct, 800))
thread_2 = threading.Thread(name="乙", target=draw, args=(acct, 800))

thread_1.start()
thread_2.start()


############ lock #############


class AccountWithLock:
    def __init__(self, account_no: Account, balance) -> None:
        self.account_no = account_no
        self._balance = balance
        self.lock = threading.RLock()

    @property
    def balance(self):
        return self._balance

    @balance.setter
    def balance(self, new_balance):
        self._balance = new_balance

    @balance.deleter
    def balance(self, new_balance):
        del self._balance

    def draw(self, draw_amount):
        self.lock.acquire()
        # 并发线程在任意时刻只有一个线程可以进入修改共享资源的代码区（也被称为临界区），所以在同一时刻最多只有一个线程处于临界区内，从而保证了线程安全。
        try:
            if self._balance >= draw_amount:
                print(f"{threading.current_thread().name}取钱成功，取出{draw_amount}")
                time.sleep(0.001)
                self._balance -= draw_amount
                print(f"余额为{self._balance}")
            else:
                print(f"{threading.current_thread().name}取钱失败，余额不足。")
        finally:
            self.lock.release()


def draw_with_lock(account: AccountWithLock, draw_amount):
    account.draw(draw_amount)


def test_lock():
    acct = AccountWithLock("123", 1000)
    thread_1 = threading.Thread(target=draw_with_lock, name="甲", args=(acct, 800))
    thread_2 = threading.Thread(target=draw_with_lock, name="乙", args=(acct, 800))
    thread_1.start()
    thread_2.start()


def test_property():
    acct = AccountWithLock("123", 1000)
    print(f"acct balance:\t {acct.balance}")
    acct.balance = 2000
    print(f"acct new balance:\t {acct.balance}")
    del acct.balance
    print(f"delete balance:\t {acct.balance}")


############ dead lock #############

"""
当两个线程相互等待对方释放资源时，就会发生死锁。Python 解释器没有监测，也不会主动采取措施来处理死锁情况，所以在进行多线程编程时应该采取措施避免出现死锁。
一旦出现死锁，整个程序既不会发生任何异常，也不会给出任何提示，只是所有线程都处于阻塞状态，无法继续。
"""


class DeadLockTester_A:
    def __init__(self) -> None:
        self.lock = threading.RLock()

    def get_A_lock(self, other_instance):
        try:
            self.lock.acquire()
            print(
                f'current thread: {threading.current_thread().name} has entered "get lock" method of A.'
            )
            time.sleep(0.2)
            print(
                f'current thread: {threading.current_thread().name} is tring to call "last" method of B.'
            )
            other_instance.last()
        finally:
            self.lock.release()

    def last(self):
        try:
            self.lock.acquire()
            print('inner the "last" method of A.')
        finally:
            self.lock.release()


class DeadLockTester_B:
    def __init__(self) -> None:
        self.lock = threading.RLock()

    def get_B_lock(self, other_instance):
        try:
            self.lock.acquire()
            print(
                f'current thread: {threading.current_thread().name} has entered "get lock" method of B.'
            )
            time.sleep(0.2)
            print(
                f'current thread: {threading.current_thread().name} is tring to call "last" method of A.'
            )
            other_instance.last()
        finally:
            self.lock.release()

    def last(self):
        try:
            self.lock.acquire()
            print('inner the "last" method of B.')
        finally:
            self.lock.release()


def test_dead_lock():
    a = DeadLockTester_A()
    b = DeadLockTester_B()

    def init():
        threading.current_thread().name = "main thread"
        a.get_A_lock(b)
        print("after enter the main thread.")

    def action():
        threading.current_thread().name = "thread_1"
        b.get_B_lock(a)
        print("after enter the thread_1")

    def test():
        threading.Thread(
            target=action
        ).start()  # this thread will propose B.lock and A.lock
        init()

    test()


if __name__ == "__main__":

    # test_lock()

    # test_property()

    # test_dead_lock()

    pass
