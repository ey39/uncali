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

import numpy as np
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
import rclpy

def matrix_to_transform_stamped(matrix: np.ndarray,
                                 parent_frame: str = "world",
                                 child_frame: str = "tool",
                                 stamp=None) -> TransformStamped:
    """
    将4x4变换矩阵转换为TransformStamped消息（使用scipy）

    参数:
        matrix (np.ndarray): 4x4齐次变换矩阵
        parent_frame (str): 父坐标系名
        child_frame (str): 子坐标系名
        stamp (builtin_interfaces.msg.Time): 时间戳，不填则使用当前 ROS 时间

    返回:
        geometry_msgs.msg.TransformStamped
    """
    assert matrix.shape == (4, 4), "输入必须是4x4的变换矩阵"

    # 提取平移
    translation = matrix[:3, 3]

    # 提取旋转
    rot_matrix = matrix[:3, :3]
    quat = R.from_matrix(rot_matrix).as_quat()  # 返回顺序为 [x, y, z, w]

    # 构建 TransformStamped
    t = TransformStamped()
    t.header.frame_id = parent_frame
    t.child_frame_id = child_frame
    t.header.stamp = stamp if stamp else rclpy.clock.Clock().now().to_msg()

    t.transform.translation.x = float(translation[0])
    t.transform.translation.y = float(translation[1])
    t.transform.translation.z = float(translation[2])

    t.transform.rotation.x = float(quat[0])
    t.transform.rotation.y = float(quat[1])
    t.transform.rotation.z = float(quat[2])
    t.transform.rotation.w = float(quat[3])

    return t


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

