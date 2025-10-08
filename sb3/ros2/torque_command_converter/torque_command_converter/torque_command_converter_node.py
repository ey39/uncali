import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
import threading
import numpy as np
import pinocchio
import time
import asyncio
import math

class RealWorldUR5e():
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
    state_topic = '/scaled_pos_joint_traj_controller/state'
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
        asyncio.ensure_future(self.pub_task())
    
    def sub_callback(self, msg):
        # msg has type: JointTrajectoryControllerState
        actual_pos = {}
        for i in range(len(msg.joint_names)):
            joint_name = msg.joint_names[i]
            joint_pos = msg.actual.positions[i]
            actual_pos[joint_name] = joint_pos
        self.current_pos = actual_pos
        if self.verbose:
            print(f'(sub) {actual_pos}')
    
    async def pub_task(self):
        while not rclpy.ok():
            await asyncio.sleep(1.0 / self.pub_freq)
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
            point.time_from_start = rclpy.duration.Duration(seconds=max(dur))
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

class TorqueCommandConverter(Node):
    def __init__(self):
        super().__init__('torque_command_converter')
        self.get_logger().info('Torque Command Converter Node has been initialized.')

        # parameters
        self.declare_parameter('torque_topic', '/torque_command')                                           # 接收 torque 的话题
        self.declare_parameter('joint_state_topic', '/joint_states')                                        # 订阅 joint_states 的话题
        self.declare_parameter('joint_trajectory_topic', '/scaled_joint_trajectory_controller/joint_trajectory')   # 发布 trajectory 的话题
        self.declare_parameter('urdf_param', 'robot_description')                                           # 或者可以是 urdf 路径
        self.declare_parameter('integration_dt', 0.02)                                                      # 积分步长(s)
        self.declare_parameter('horizon', 0.02)                                                             # 预测时长(s) (单点）
        self.declare_parameter('tau_regularization', 1e-8)                                                  # 数值稳定项
        self.declare_parameter('publish_rate', 50.0)                                                        # 发布频率 Hz
        self.declare_parameter('joint_names', [])                                                           # 可选：按序的关节名（覆盖从 joint_states 读取顺序）

        self.torque_topic = self.get_parameter('torque_topic').get_parameter_value().string_value
        self.joint_state_topic = self.get_parameter('joint_state_topic').get_parameter_value().string_value
        self.joint_trajectory_topic = self.get_parameter('joint_trajectory_topic').get_parameter_value().string_value
        self.urdf_param = self.get_parameter('urdf_param').get_parameter_value().string_value
        self.dt = float(self.get_parameter('integration_dt').get_parameter_value().double_value)
        self.horizon = float(self.get_parameter('horizon').get_parameter_value().double_value)
        self.tau_reg = float(self.get_parameter('tau_regularization').get_parameter_value().double_value)
        self.publish_rate = float(self.get_parameter('publish_rate').get_parameter_value().double_value)
        self.joint_names_param = self.get_parameter('joint_names').get_parameter_value().string_array_value

        self.get_logger().info(f'Node params: torque_topic={self.torque_topic}, joint_traj_topic={self.joint_trajectory_topic}, dt={self.dt}, horizon={self.horizon}')

        # Pinocchio 模型加载：尝试从 param server 读取 robot_description，否则假定参数是路径
        urdf_string = None
        try:
            urdf_string = self.get_parameter(self.urdf_param).get_parameter_value().string_value
        except Exception:
            # parameter may not be declared as param containing URDF string; attempt to read directly from param server
            try:
                urdf_string = self.get_parameter('robot_description').get_parameter_value().string_value
            except Exception:
                urdf_string = None

        if urdf_string is None or len(urdf_string) < 20:
            # 你可以把 urdf 文件路径放到另一个 ROS 参数如 'urdf_path'，这里尝试读取它
            try:
                urdf_path = self.get_parameter('urdf_path').get_parameter_value().string_value
                self.robot = pinocchio.robot_wrapper.RobotWrapper.BuildFromURDF(urdf_path, ["/"], pinocchio.JointModelFreeFlyer())
            except Exception as e:
                self.get_logger().error("无法从参数读取 URDF。请在节点参数中提供有效的 'robot_description' 字符串或 'urdf_path' 文件路径。")
                raise e
        else:
            # 从 URDF 字符串构造 RobotWrapper
            try:
                # 注意：RobotWrapper.BuildFromXML 期望 xml 字符串
                self.robot = pinocchio.robot_wrapper.RobotWrapper.BuildFromXML(urdf_string, ["/"])
            except Exception as e:
                self.get_logger().error("用 BuildFromXML 加载 URDF 失败，请确认 URDF 字符串或使用 urdf_path。")
                raise e

        self.model = self.robot.model
        self.data = self.model.createData()

        # 存储当前 joint state（按 joint_names 顺序）
        self.current_q = None
        self.current_dq = None
        self.joint_names = None

        # 订阅/发布
        cb_group = ReentrantCallbackGroup()
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.joint_state_topic,
            self.joint_state_cb,
            10,
            callback_group=cb_group
        )
        self.torque_sub = self.create_subscription(
            Float64MultiArray,
            self.torque_topic,
            self.torque_cb,
            10,
            callback_group=cb_group
        )

        # self.traj_pub = self.create_publisher(JointTrajectory, self.joint_trajectory_topic, 10)

        # 缓存最后收到的 torque
        self.last_tau = None
        self.tau_lock = threading.Lock()

        # 启动周期发布线程（定期根据最近的 torque 与 jointstate 生成 trajectory）
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_cb)

    def joint_state_cb(self, msg: JointState):
        # 记录 joint_names, q, dq。保持按指定 joint order
        if self.joint_names is None:
            # 优先使用参数中的 joint_names
            if len(self.joint_names_param) > 0:
                self.joint_names = list(self.joint_names_param)
            else:
                self.joint_names = list(msg.name)

            self.get_logger().info(f'Using joint names: {self.joint_names}')

        # 将 msg 里的 q/dq 对应到 self.joint_names 的顺序（填 0 若缺失）
        n = len(self.joint_names)
        q = np.zeros(n)
        dq = np.zeros(n)
        name_to_index = {name: i for i, name in enumerate(msg.name)}
        for i, jn in enumerate(self.joint_names):
            if jn in name_to_index:
                idx = name_to_index[jn]
                if len(msg.position) > idx:
                    q[i] = msg.position[idx]
                if len(msg.velocity) > idx:
                    dq[i] = msg.velocity[idx]
        self.current_q = q
        self.current_dq = dq

    def torque_cb(self, msg: Float64MultiArray):
        # 假设 msg.data 长度等于关节数，按 self.joint_names 顺序
        with self.tau_lock:
            self.last_tau = np.array(msg.data, dtype=float)

    def timer_cb(self):
        # 周期性产生 trajectory 并发布
        if self.current_q is None or self.current_dq is None:
            # 尚未收到 joint_states
            return

        with self.tau_lock:
            tau = None if self.last_tau is None else self.last_tau.copy()

        if tau is None:
            # 没有新的 torque 指令，可以选择不发布或者 hold position（这里不发布）
            return

        q = self.current_q.copy()
        dq = self.current_dq.copy()
        n = len(q)

        # 校验尺寸
        if tau.shape[0] != n:
            self.get_logger().warn(f"收到 torque 长度 {tau.shape[0]} 与 joints 长度 {n} 不匹配。忽略该消息。")
            return

        # 计算动力学项：M(q) 和 nonlinear terms (C+g)
        # Pinocchio: 先计算 CRBA（M），然后计算 non-linear effects via computeAllTerms
        try:
            # compute CRBA (mass matrix)
            pinocchio.crba(self.model, self.data, q)
            M = self.data.M.copy()

            # compute non-linear terms (Coriolis + gravity)
            pinocchio.computeAllTerms(self.model, self.data, q, dq)
            nle = self.data.nle.copy()  # nonlinear effects
        except Exception as e:
            self.get_logger().error(f"Pinocchio 动力学计算失败: {e}")
            return

        # 数值保护：确保 M 正定（对角上加小量）
        M_reg = M + np.eye(n) * self.tau_reg

        # ddq = M^{-1} (tau - nle)
        rhs = tau - nle
        try:
            ddq = np.linalg.solve(M_reg, rhs)
        except np.linalg.LinAlgError:
            self.get_logger().warn("质量矩阵求逆失败，尝试伪逆")
            ddq = np.linalg.pinv(M_reg).dot(rhs)

        # 简单积分得到期望 velocity 与 position（单步）
        dt = self.dt
        dq_des = dq + ddq * dt
        q_des = q + dq_des * dt

        # 生成 JointTrajectory（当前 pos -> q_des，时间为 horizon）
        traj = JointTrajectory()
        traj.joint_names = list(self.joint_names)

        # 当前点（可选：为 controller 指定起点，这不是必须，但有时 helpful）
        p0 = JointTrajectoryPoint()
        p0.positions = q.tolist()
        p0.velocities = dq.tolist()
        p0.accelerations = (ddq * 0.0).tolist()  # 未提供当前加速度
        p0.time_from_start = rclpy.duration.Duration(seconds=0.0).to_msg()

        p1 = JointTrajectoryPoint()
        p1.positions = q_des.tolist()
        p1.velocities = dq_des.tolist()
        p1.accelerations = (ddq).tolist()
        # time_from_start: 设为 horizon
        tfs = self.horizon
        # 如果 horizon 太小，确保至少是 dt
        if tfs <= 0.0:
            tfs = dt
        p1.time_from_start = rclpy.duration.Duration(seconds=float(tfs)).to_msg()

        traj.points = [p0, p1]

        # 发布
        self.traj_pub.publish(traj)
        # （可选）清除 last_tau 或继续使用，视你想要的控制语义而定
        # with self.tau_lock:
        #     self.last_tau = None


def main(args=None):
    rclpy.init(args=args)
    node = TorqueCommandConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()