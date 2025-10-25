import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from builtin_interfaces.msg import Duration
from common.one_euro_filter import OneEuroFilter
import numpy as np
import asyncio
import math

class RealWorldUR5e(Node):
    # Defined in ur10.usd
    sim_dof_angle_limits = [
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
    ] 
    pi = math.pi
    servo_angle_limits = [
        (-2*pi, 2*pi),
        (-2*pi, 2*pi),
        (-2*pi, 2*pi),
        (-2*pi, 2*pi),
        (-2*pi, 2*pi),
        (-2*pi, 2*pi),
    ]
    # ROS-related strings
    state_topic = '/scaled_joint_trajectory_controller/state'
    cmd_topic = '/scaled_joint_trajectory_controller/joint_trajectory'
    joint_names = [
        'elbow_joint',
        'shoulder_lift_joint',
        'shoulder_pan_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint'
    ]
    # Joint name mapping to simulation action index
    joint_name_to_idx = {
        'elbow_joint': 2,
        'shoulder_lift_joint': 1,
        'shoulder_pan_joint': 0,
        'wrist_1_joint': 3,
        'wrist_2_joint': 4,
        'wrist_3_joint': 5
    }

    def __init__(self, fail_quietely=False, verbose=False) -> None:
        super().__init__("RealWorldUR5e")
        print("Connecting to real-world UR5e")
        self.fail_quietely = fail_quietely
        self.verbose = verbose
        self.pub_freq = 10 # Hz
        # Not really sure if current_pos and target_pos require mutex here.
        self.current_pos = None
        self.target_pos = None

        self.freq = 60.0
        beta = 0.01
        min_cutoff = 1.0
        d_cutoff = 1.0
        self.filters = [OneEuroFilter(x0=0, t0=0, dx0=0, beta=beta, min_cutoff=min_cutoff, d_cutoff=d_cutoff) for _ in range(6)]
        self.time = 0
        
        if self.verbose:
            print("Receiving real-world UR10 joint angles...")
            print("If you didn't see any outputs, you may have set up UR5 or ROS incorrectly.")

        self.get_logger().info("Node has already been initialized, do nothing")
        self.sub = self.create_subscription(
            JointTrajectoryControllerState,
            self.state_topic,
            self.sub_callback,
            1
        )
        
        self.pub = self.create_publisher(
            JointTrajectory,
            self.cmd_topic,
            1
        ) 
        # self.min_traj_dur = 5.0 / self.pub_freq  # Minimum trajectory duration
        self.min_traj_dur = 0  # Minimum trajectory duration

        # For catching exceptions in asyncio
        def custom_exception_handler(loop, context):
            print(context)
        # Ref: https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.set_exception_handler
        asyncio.get_event_loop().set_exception_handler(custom_exception_handler)
        # Ref: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_ros_custom_message.html
        asyncio.ensure_future(self.pub_task()) # 在事件循环中启动一个新的协程任务
    
    def sub_callback(self, msg):
        # msg has type: JointTrajectoryControllerState
        self.time += 1 / self.freq
        actual_pos = {}
        for i in range(len(msg.joint_names)):
            joint_name = msg.joint_names[i]
            joint_pos = msg.actual.positions[i]
            # filter
            # joint_pos = self.filters[i](self.time, joint_pos)
            actual_pos[joint_name] = joint_pos
        self.current_pos = actual_pos
        # if self.verbose:
        #     print(f'(sub) {actual_pos}')
    
    async def pub_task(self):
        while rclpy.ok():
            await asyncio.sleep(1.0 / self.pub_freq)
            # print("pub_task running...")
            if self.current_pos is None:
                # Not ready (recieved UR state) yet
                continue
            if self.target_pos is None:
                # No command yet
                continue
            # Construct message
            dur = [] # move duration of each joints
            traj = JointTrajectory()
            traj.joint_names = self.joint_names
            point = JointTrajectoryPoint()
            moving_average = 1
            for name in traj.joint_names:
                pos = self.current_pos[name]
                cmd = pos * (1-moving_average) + self.target_pos[self.joint_name_to_idx[name]] * moving_average
                max_vel = 3.15 # from ur5.urdf (or ur5.urdf.xacro)
                duration = abs(cmd - pos) / max_vel # time = distance / velocity
                dur.append(max(duration, self.min_traj_dur))
                point.positions.append(cmd)
            # point.time_from_start = rospy.Duration(max(dur))
            point.time_from_start = Duration(sec=int(max(dur)), nanosec=int((max(dur) % 1) * 1e9))
            # point.time_from_start = rclpy.duration.Duration(seconds=max(dur))
            traj.points.append(point)
            self.pub.publish(traj)
            print(f'(pub) {point.positions}')

    def send_joint_pos(self, joint_pos):
        if len(joint_pos) != 6:
            raise Exception("The length of UR10 joint_pos is {}, but should be 6!".format(len(joint_pos)))

        # Convert Sim angles to Real angles
        target_pos = [0] * 6
        for i, pos in enumerate(joint_pos):
            if i == 5:
                # Ignore the gripper joints for Reacher task
                continue
            # Map [L, U] to [A, B]
            L, U, inversed = self.sim_dof_angle_limits[i]
            A, B = self.servo_angle_limits[i]
            angle = np.rad2deg(float(pos))
            if not L <= angle <= U:
                print("The {}-th simulation joint angle ({}) is out of range! Should be in [{}, {}]".format(i, angle, L, U))
                angle = np.clip(angle, L, U)
            target_pos[i] = (angle - L) * ((B-A)/(U-L)) + A # Map [L, U] to [A, B]
            if inversed:
                target_pos[i] = (B-A) - (target_pos[i] - A) + A # Map [A, B] to [B, A]
            if not A <= target_pos[i] <= B:
                raise Exception("(Should Not Happen) The {}-th real world joint angle ({}) is out of range! hould be in [{}, {}]".format(i, target_pos[i], A, B))
            self.target_pos = target_pos

if __name__ == "__main__":
    print("Make sure you are running `roslaunch ur_robot_driver`.")
    print("If the machine running Isaac is not the ROS master node, " + \
          "make sure you have set the environment variables: " + \
          "`ROS_MASTER_URI` and `ROS_HOSTNAME`/`ROS_IP` correctly.")
    rclpy.init(args=args)
    ur5e = RealWorldUR5e(verbose=True)
    rclpy.spin(ur5e) # 保持节点运行，检测是否收到退出指令（Ctrl+C）
    rclpy.shutdown() # 关闭rclpy