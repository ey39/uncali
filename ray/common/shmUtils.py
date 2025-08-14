from multiprocessing import shared_memory
import pickle, time
from multiprocessing import Lock

class SharedMemoryChannel:
    def __init__(self, name, size=4096):
        self.name = name
        self.size = size
        self.lock = Lock()
        try:
            # 尝试连接已存在的共享内存
            self.shm = shared_memory.SharedMemory(name=name)
            self.is_owner = False
            print(f"[{name}] 共享内存已存在，连接成功")
        except FileNotFoundError:
            # 不存在就创建
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=size)
            self.shm.buf[:1] = b'\x00'  # 标志位清空
            self.is_owner = True
            print(f"[{name}] 共享内存不存在，创建成功")

    def send(self, data):
        with self.lock:
            payload = pickle.dumps(data)
            if len(payload) > self.size - 1:
                raise ValueError("数据太大，超出共享内存大小")
            self.shm.buf[1:1+len(payload)] = payload
            self.shm.buf[0] = 1

    def recv(self, timeout=None):
        start = time.time()
        while True:
            if self.shm.buf[0] == 1:
                with self.lock:
                    raw = bytes(self.shm.buf[1:])
                    obj = pickle.loads(raw.rstrip(b'\x00'))
                    self.shm.buf[0] = 0
                    return obj
            time.sleep(0.01)
            if timeout and time.time() - start > timeout:
                break

    def close(self):
        self.shm.close()

    def unlink(self):
        if self.is_owner:
            self.shm.unlink()

def shared_memory_exists(name: str) -> bool:
    try:
        shm = shared_memory.SharedMemory(name=name, create=False)
        shm.close()  # 记得关闭句柄
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"Unexpected error when checking shared memory: {e}")
        return False

if __name__ == '__main__':
    '''
    recv
    '''
    # channel = SharedMemoryChannel("chatbus")
    # while True:
    #     msg = channel.recv(timeout=5)
    #     print("收到：", msg)
    '''
    send
    '''
    # import time
    # channel = SharedMemoryChannel("chatbus")
    # for i in range(100):
    #     channel.send({"from": "sender", "msg": f"hello {i}"})
    #     time.sleep(1)
    # channel.close()

