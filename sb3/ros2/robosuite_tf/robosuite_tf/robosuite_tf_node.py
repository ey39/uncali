import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
import time
from .shmUtils import SharedMemoryChannel, matrix_to_transform_stamped


def main(args=None):
    rclpy.init(args=args) 
    node = Node("robosuite_tf_node")  
    node.declare_parameter('channel', 'chatbus_0')

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    channelstr = node.get_parameter('channel').get_parameter_value().string_value
    channel = SharedMemoryChannel(channelstr)
    static_tf_pub = StaticTransformBroadcaster(node)

    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1) 
            msg = channel.recv(timeout=1)
            if msg is not None:
                for key, value in msg.items():
                    key_items = key.split("_")
                    # print(key_items)
                    t = matrix_to_transform_stamped(
                        matrix=value,
                        parent_frame=key_items[2],
                        child_frame=key_items[1],
                    )
                    # print(t)
                    static_tf_pub.sendTransform(t)

            time.sleep(0.01)
    finally:
        node.destroy_node()
        rclpy.shutdown()