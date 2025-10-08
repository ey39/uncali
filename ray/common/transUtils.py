import numpy as np
from scipy.spatial.transform import Rotation
from typing import List, Union, Tuple
import math

def combined_reward_scalar(x, x_ref=0.5, K=2.0, R=10.0, x_c=0.8, eps=1e-12):
    """
    Hybrid reward function:
        - log-shaped for x <= x_c
        - linear extension for x > x_c
    """
    val_at_xc = -K * math.log10((x_c + eps) / x_ref)
    slope = -K / ((x_c + eps) * math.log(10))
    
    if x <= x_c:
        y = -K * math.log10((x + eps) / x_ref)
    else:
        y = val_at_xc + slope * (x - x_c)
        
    if y > R:
        return R
    if y < -R:
        return -R
    return y

def combined_reward_array(x, x_ref=0.5, K=2.0, R=10.0, x_c=0.8, eps=1e-12):
    """
    Hybrid reward function:
        - log-shaped for x <= x_c
        - linear extension for x > x_c
    """
    x = np.array(x)
    y = np.empty_like(x)

    val_at_xc = -K * np.log10((x_c + eps) / x_ref)
    slope = -K / ((x_c + eps) * np.log(10))
    linear = val_at_xc + slope * (x - x_c)
    mask = (x <= x_c)
    y[mask] = -K * np.log10((x[mask] + eps) / x_ref)
    y[~mask] = linear[~mask]

    return np.clip(y, -R, R)

def combined_reward(x, *args, **kwargs):
    if isinstance(x, (list, tuple, np.ndarray)):
        return combined_reward_array(np.array(x), *args, **kwargs)
    else:
        return combined_reward_scalar(float(x), *args, **kwargs)

def invert_homogeneous_matrix(T):
    """
    计算齐次变换矩阵 T 的逆（4x4）
    """
    R = T[:3, :3]
    t = T[:3, 3]

    R_inv = R.T
    t_inv = -R_inv @ t

    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv

    return T_inv

def generate_random_homogeneous_transform_shell(
    translation_radius=(0.05, 0.1),  # (R_min, R_max) 球壳范围
    rotation_mode='axis_angle',
    rotation_range=(0, 10),  # 单位：度
    seed=None
):
    """
    生成一个随机的4x4齐次变换矩阵（在球壳内随机采样）

    Parameters
    ----------
    translation_radius : tuple
        (R_min, R_max)，采样范围 (m)，保证 R_min <= ||t|| <= R_max
    rotation_mode : str
        姿态扰动方式:
        - 'random': 完全随机旋转 (SO(3))
        - 'axis_angle': 在给定角度范围内随机旋转
        - 'euler': 欧拉角扰动
    rotation_range : tuple, optional
        旋转范围（度），依赖 rotation_mode
    seed : int, optional
        随机种子

    Returns
    -------
    numpy.ndarray
        4x4齐次变换矩阵
    """
    if seed is not None:
        np.random.seed(seed)

    # --- 平移：球壳内随机采样 ---
    r_min, r_max = translation_radius
    direction = np.random.randn(3)
    direction /= np.linalg.norm(direction)
    radius = np.random.uniform(r_min, r_max)  # 保证在球壳内
    translation = direction * radius

    # --- 姿态 ---
    if rotation_mode == 'random':
        rotation = Rotation.random().as_matrix()

    elif rotation_mode == 'axis_angle':
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        angle_deg = np.random.uniform(rotation_range[0], rotation_range[1])
        angle_rad = np.radians(angle_deg)
        rotation = Rotation.from_rotvec(axis * angle_rad).as_matrix()

    elif rotation_mode == 'euler':
        angles = np.random.uniform(rotation_range[0], rotation_range[1], 3)
        rotation = Rotation.from_euler('xyz', np.radians(angles)).as_matrix()

    else:
        raise ValueError(f"不支持的旋转模式: {rotation_mode}")

    # --- 齐次矩阵 ---
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation
    transform_matrix[:3, 3] = translation

    return transform_matrix

def generate_random_homogeneous_transform(
    translation_range=(-1.0, 1.0),
    rotation_mode='random',
    rotation_range=None,
    seed=None
):
    """
    生成一个随机的4x4齐次变换矩阵
    
    Parameters:
    -----------
    translation_range : tuple or list
        平移范围，可以是：
        - (min, max): 所有轴使用相同范围
        - [(x_min, x_max), (y_min, y_max), (z_min, z_max)]: 各轴独立范围
    
    rotation_mode : str
        旋转生成模式：
        - 'random': 完全随机旋转
        - 'euler': 使用欧拉角生成，需要配合rotation_range参数
        - 'axis_angle': 使用轴角表示，需要配合rotation_range参数
    
    rotation_range : tuple or list, optional
        旋转范围（仅当rotation_mode不为'random'时使用）：
        - 对于'euler'模式: (min_angle, max_angle) 单位为度
        - 对于'axis_angle'模式: (min_angle, max_angle) 单位为度
    
    seed : int, optional
        随机种子
    
    Returns:
    --------
    numpy.ndarray
        4x4齐次变换矩阵
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # 生成随机平移向量
    if isinstance(translation_range[0], (int, float)):
        # 统一范围
        t_min, t_max = translation_range
        translation = np.random.uniform(t_min, t_max, 3)
    else:
        # 各轴独立范围
        translation = np.array([
            np.random.uniform(translation_range[0][0], translation_range[0][1]),
            np.random.uniform(translation_range[1][0], translation_range[1][1]),
            np.random.uniform(translation_range[2][0], translation_range[2][1])
        ])
    
    # 生成随机旋转矩阵
    if rotation_mode == 'random':
        # 完全随机旋转
        rotation = Rotation.random().as_matrix()
    
    elif rotation_mode == 'euler':
        if rotation_range is None:
            rotation_range = (-180, 180)
        
        # 使用欧拉角生成旋转
        angles = np.random.uniform(
            rotation_range[0], rotation_range[1], 3
        )
        rotation = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
    
    elif rotation_mode == 'axis_angle':
        if rotation_range is None:
            rotation_range = (-180, 180)
        
        # 随机轴方向
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        
        # 随机角度
        angle = np.random.uniform(rotation_range[0], rotation_range[1])
        rotation = Rotation.from_rotvec(
            axis * np.radians(angle)
        ).as_matrix()
    
    else:
        raise ValueError(f"不支持的旋转模式: {rotation_mode}")
    
    # 构造齐次变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation
    transform_matrix[:3, 3] = translation
    
    return transform_matrix


def generate_perturbed_transform(
    base_transform,
    translation_error_range=(-0.01, 0.01),
    rotation_error_range=(-1.0, 1.0),  # 单位：度
    rotation_mode='euler',  # 'euler' 或 'axis_angle'
    seed=None
):
    """
    基于已有的齐次变换矩阵，生成一个扰动后的新变换，旋转和平移误差可控。
    
    Parameters:
    -----------
    base_transform : numpy.ndarray
        原始的 4x4 齐次变换矩阵
        
    translation_error_range : tuple or list
        平移扰动范围（单位：米），可以是：
        - (min, max)：所有轴相同范围
        - [(x_min, x_max), (y_min, y_max), (z_min, z_max)]：各轴独立范围
    
    rotation_error_range : tuple
        旋转误差范围（单位：度），旋转角度限制
        
    rotation_mode : str
        旋转扰动的生成方式：
        - 'euler': 使用 xyz 欧拉角扰动
        - 'axis_angle': 使用轴角扰动
        
    seed : int, optional
        随机种子，保证复现性
    
    Returns:
    --------
    numpy.ndarray
        扰动后的 4x4 齐次变换矩阵
    """
    if seed is not None:
        np.random.seed(seed)

    # 提取原始旋转和平移
    R_base = base_transform[:3, :3]
    t_base = base_transform[:3, 3]

    # 平移扰动
    if isinstance(translation_error_range[0], (int, float)):
        t_min, t_max = translation_error_range
        delta_t = np.random.uniform(t_min, t_max, 3)
    else:
        delta_t = np.array([
            np.random.uniform(*translation_error_range[0]),
            np.random.uniform(*translation_error_range[1]),
            np.random.uniform(*translation_error_range[2])
        ])
    t_new = t_base + delta_t

    # 旋转扰动
    if rotation_mode == 'euler':
        angles = np.random.uniform(
            rotation_error_range[0], rotation_error_range[1], 3
        )
        delta_R = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
    elif rotation_mode == 'axis_angle':
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        angle = np.random.uniform(*rotation_error_range)
        delta_R = Rotation.from_rotvec(axis * np.radians(angle)).as_matrix()
    else:
        raise ValueError(f"不支持的旋转模式: {rotation_mode}")

    R_new = delta_R @ R_base  # 右扰动模型

    # 构造新的齐次变换矩阵
    T_new = np.eye(4)
    T_new[:3, :3] = R_new
    T_new[:3, 3] = t_new
    return T_new

def calculate_pos_error(t1, t2):
    translation_error = t2 - t1
    translation_magnitude = np.linalg.norm(translation_error)
    return translation_magnitude

def calculate_rot_error(r1, r2, angle_unit='degrees'):
    R1 = Rotation.from_rotvec(r1).as_matrix()
    R2 = Rotation.from_rotvec(r2).as_matrix()
    R_error = R2 @ R1.T
    
    # 使用scipy计算旋转误差
    rot_error = Rotation.from_matrix(R_error)
    
    # 提取旋转误差的轴角表示
    rotvec = rot_error.as_rotvec()
    rotation_error_angle = np.linalg.norm(rotvec)
    
    # 转换角度单位
    if angle_unit == 'degrees':
        rotation_error_angle = np.degrees(rotation_error_angle)
    return rotation_error_angle

def calculate_pose_error(T1, T2, angle_unit='degrees'):
    """
    计算两个齐次变换矩阵之间的位姿误差
    
    Parameters:
    -----------
    T1 : numpy.ndarray
        第一个4x4齐次变换矩阵 (物体1的位姿)
    T2 : numpy.ndarray
        第二个4x4齐次变换矩阵 (物体2的位姿)
    angle_unit : str
        角度单位，'degrees' 或 'radians'
    
    Returns:
    --------
    dict
        包含误差信息的字典：
        - 'translation_error': 平移误差向量 (3,)
        - 'translation_magnitude': 平移误差的模长
        - 'rotation_error_matrix': 旋转误差矩阵 (3x3)
        - 'rotation_error_angle': 旋转误差角度（标量）
        - 'rotation_error_axis': 旋转误差轴向量 (3,)
        - 'rotation_error_euler': 旋转误差的欧拉角 (3,)
    """
    
    # 提取平移向量
    t1 = T1[:3, 3]
    t2 = T2[:3, 3]
    
    # 提取旋转矩阵
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    
    # 计算平移误差
    translation_error = t2 - t1
    translation_magnitude = np.linalg.norm(translation_error)
    
    # 计算旋转误差
    # 相对旋转矩阵: R_error = R2 * R1^T
    R_error = R2 @ R1.T
    
    # 使用scipy计算旋转误差
    rot_error = Rotation.from_matrix(R_error)
    
    # 提取旋转误差的轴角表示
    rotvec = rot_error.as_rotvec()
    rotation_error_angle = np.linalg.norm(rotvec)
    
    if rotation_error_angle > 1e-6:
        rotation_error_axis = rotvec / rotation_error_angle
    else:
        rotation_error_axis = np.array([0, 0, 1])  # 默认轴向量
    
    # 转换角度单位
    if angle_unit == 'degrees':
        rotation_error_angle = np.degrees(rotation_error_angle)
        rotation_error_euler = rot_error.as_euler('xyz', degrees=True)
    else:
        rotation_error_euler = rot_error.as_euler('xyz', degrees=False)
    
    return {
        'translation_error': translation_error,
        'translation_magnitude': translation_magnitude,
        'rotation_error_matrix': R_error,
        'rotation_error_angle': rotation_error_angle,
        'rotation_error_axis': rotation_error_axis,
        'rotation_error_euler': rotation_error_euler
    }

def compose_transforms(*transforms):
    """
    将多个齐次变换矩阵进行复合变换
    
    Parameters:
    -----------
    *transforms : numpy.ndarray
        可变数量的4x4齐次变换矩阵
        变换顺序：T_final = T_n @ T_{n-1} @ ... @ T_2 @ T_1
    
    Returns:
    --------
    numpy.ndarray
        复合后的4x4齐次变换矩阵
    """
    if len(transforms) == 0:
        return np.eye(4)
    
    result = transforms[0].copy()
    for T in transforms[1:]:
        result = result @ T
    
    return result

def compose_transforms_list(transform_list: List[np.ndarray]):
    """
    将齐次变换矩阵列表进行复合变换
    
    Parameters:
    -----------
    transform_list : List[numpy.ndarray]
        齐次变换矩阵列表
    
    Returns:
    --------
    numpy.ndarray
        复合后的4x4齐次变换矩阵
    """
    if len(transform_list) == 0:
        return np.eye(4)
    
    result = transform_list[0].copy()
    for T in transform_list[1:]:
        result = result @ T
    
    return result

def quaternion_to_homogeneous_matrix(quaternion, translation=None):
    """
    将四元数（和可选的平移）转换为齐次变换矩阵
    
    Parameters:
    -----------
    quaternion : array-like
        四元数 [x, y, z, w] 或 [w, x, y, z]，会自动检测格式
    translation : array-like, optional
        平移向量 [x, y, z]，默认为零向量
    
    Returns:
    --------
    numpy.ndarray
        4x4齐次变换矩阵
    """
    quaternion = np.array(quaternion, dtype=float)
    
    if translation is None:
        translation = np.zeros(3)
    else:
        translation = np.array(translation, dtype=float)
    
    # 检测四元数格式并标准化
    quaternion = normalize_quaternion(quaternion)
    
    # [x,y,z,w] 格式
    rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
    
    # 构造齐次变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation
    
    return transform_matrix


def homogeneous_matrix_to_quaternion(transform_matrix, quaternion_format='xyzw'):
    """
    将齐次变换矩阵转换为四元数和平移向量
    
    Parameters:
    -----------
    transform_matrix : numpy.ndarray
        4x4齐次变换矩阵
    quaternion_format : str
        输出四元数格式：'xyzw' 或 'wxyz'
    
    Returns:
    --------
    tuple
        (quaternion, translation) 其中：
        - quaternion: 四元数 [x,y,z,w] 或 [w,x,y,z]
        - translation: 平移向量 [x,y,z]
    """
    transform_matrix = np.array(transform_matrix)
    
    # 提取旋转矩阵和平移向量
    rotation_matrix = transform_matrix[:3, :3]
    translation = transform_matrix[:3, 3]
    
    # 转换为四元数
    rotation = Rotation.from_matrix(rotation_matrix)
    quaternion_xyzw = rotation.as_quat()  # scipy默认返回[x,y,z,w]格式
    
    if quaternion_format == 'wxyz':
        quaternion = np.array([quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]])
    else:
        quaternion = quaternion_xyzw
    
    return quaternion, translation

def homogeneous_matrix_to_axisangle(transform_matrix):
    """
    将齐次变换矩阵转换为轴角和平移向量
    
    Parameters:
    -----------
    transform_matrix : numpy.ndarray
        4x4齐次变换矩阵
    
    Returns:
    --------
    tuple
        (axisangle, translation) 其中：
        - axisangle: 轴角 [rx, ry, rz]
        - translation: 平移向量 [x,y,z]
    """
    transform_matrix = np.array(transform_matrix)
    
    # 提取旋转矩阵和平移向量
    rotation_matrix = transform_matrix[:3, :3]
    translation = transform_matrix[:3, 3]
    
    # 转换为四元数
    rotation = Rotation.from_matrix(rotation_matrix)
    axisangle = rotation.as_rotvec()
    
    return axisangle, translation

def pose_to_homogeneous_matrix(position, quaternion):
    """
    将位置和四元数转换为齐次变换矩阵
    
    Parameters:
    -----------
    position : array-like
        位置向量 [x, y, z]
    quaternion : array-like
        四元数 [x, y, z, w] 或 [w, x, y, z]
    
    Returns:
    --------
    numpy.ndarray
        4x4齐次变换矩阵
    """
    return quaternion_to_homogeneous_matrix(quaternion, position)


def homogeneous_matrix_to_pose(transform_matrix, quaternion_format='xyzw'):
    """
    将齐次变换矩阵转换为位置和四元数
    
    Parameters:
    -----------
    transform_matrix : numpy.ndarray
        4x4齐次变换矩阵
    quaternion_format : str
        输出四元数格式：'xyzw' 或 'wxyz'
    
    Returns:
    --------
    tuple
        (position, quaternion) 其中：
        - position: 位置向量 [x,y,z]
        - quaternion: 四元数 [x,y,z,w] 或 [w,x,y,z]
    """
    quaternion, translation = homogeneous_matrix_to_quaternion(transform_matrix, quaternion_format)
    return translation, quaternion


def normalize_quaternion(quaternion):
    """
    归一化四元数
    
    Parameters:
    -----------
    quaternion : array-like
        四元数
    
    Returns:
    --------
    numpy.ndarray
        归一化后的四元数
    """
    quaternion = np.array(quaternion, dtype=float)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-8:
        # 如果四元数为零，返回单位四元数
        return np.array([0., 0., 0., 1.])
    return quaternion / norm

def is_point_in_cylinder(base_center, axis_vector, radius, height, point):
    """
    判断一个点是否在任意朝向的有限高圆柱体内

    参数:
    - base_center: np.array([x, y, z]) 圆柱底面中心
    - axis_vector: np.array([x, y, z]) 圆柱轴向（可以不是单位向量）
    - radius: float 圆柱半径
    - height: float 圆柱高度
    - point: np.array([x, y, z]) 被判断的点

    返回:
    - bool: 是否在圆柱体内
    """
    axis = axis_vector / np.linalg.norm(axis_vector)  # 单位向量
    vec_to_point = point - base_center

    # 投影在轴向上的距离（判断是否在高度范围内）
    projection_length = np.dot(vec_to_point, axis)
    if projection_length < 0 or projection_length > height:
        return False

    # 计算点到轴线的垂直距离（判断是否在半径范围内）
    projection_point = base_center + projection_length * axis
    radial_distance = np.linalg.norm(point - projection_point)
    return radial_distance <= radius

from collections import deque

class SuccessTracker:
    def __init__(self, max_len):
        self.max_len = max_len
        self.history = deque(maxlen=self.max_len)

    def add_result(self, success: bool):
        self.history.append(success)

    def success_rate(self) -> float:
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)
    
    def clear(self):
        self.history.clear()