import numpy as np
from scipy.spatial.transform import Rotation
from typing import List, Union, Tuple

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

def euler2quat(roll, pitch, yaw):
    """
    Convert euler angles to quaternion.
    Angles in radians.
    Returns (w, x, y, z)
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [w, x, y, z]

def print_dict(d, indent=0):
    for key, value in d.items():
        prefix = '  ' * indent + f"{key}: "
        if isinstance(value, dict):
            print(prefix)
            print_dict(value, indent + 1)
        else:
            print(prefix + str(value))


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

def print_transform_info(T):
    """
    打印变换矩阵的信息
    """
    print("齐次变换矩阵:")
    print(T)
    print(f"\n平移向量: {T[:3, 3]}")
    
    # 从旋转矩阵提取欧拉角
    rotation = Rotation.from_matrix(T[:3, :3])
    euler_angles = rotation.as_euler('xyz', degrees=True)
    print(f"欧拉角 (度): {euler_angles}")
    

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


def calculate_pose_error_simple(T1, T2, angle_unit='degrees'):
    """
    简化版本：只返回平移误差模长和旋转误差角度
    
    Parameters:
    -----------
    T1, T2 : numpy.ndarray
        4x4齐次变换矩阵
    angle_unit : str
        角度单位，'degrees' 或 'radians'
    
    Returns:
    --------
    tuple
        (平移误差模长, 旋转误差角度)
    """
    
    # 平移误差
    translation_error = np.linalg.norm(T2[:3, 3] - T1[:3, 3])
    
    # 旋转误差
    R_error = T2[:3, :3] @ T1[:3, :3].T
    rotation_error_angle = np.linalg.norm(Rotation.from_matrix(R_error).as_rotvec())
    
    if angle_unit == 'degrees':
        rotation_error_angle = np.degrees(rotation_error_angle)
    
    return translation_error, rotation_error_angle


def print_pose_error(error_dict, precision=4):
    """
    打印位姿误差信息
    """
    print("=== 位姿误差分析 ===")
    print(f"平移误差向量: [{error_dict['translation_error'][0]:.{precision}f}, "
          f"{error_dict['translation_error'][1]:.{precision}f}, "
          f"{error_dict['translation_error'][2]:.{precision}f}]")
    print(f"平移误差模长: {error_dict['translation_magnitude']:.{precision}f}")
    
    print(f"\n旋转误差角度: {error_dict['rotation_error_angle']:.{precision}f}°")
    print(f"旋转误差轴向量: [{error_dict['rotation_error_axis'][0]:.{precision}f}, "
          f"{error_dict['rotation_error_axis'][1]:.{precision}f}, "
          f"{error_dict['rotation_error_axis'][2]:.{precision}f}]")
    print(f"旋转误差欧拉角: [{error_dict['rotation_error_euler'][0]:.{precision}f}, "
          f"{error_dict['rotation_error_euler'][1]:.{precision}f}, "
          f"{error_dict['rotation_error_euler'][2]:.{precision}f}]°")


def batch_pose_error_analysis(T_list1, T_list2, angle_unit='degrees'):
    """
    批量计算位姿误差，用于统计分析
    
    Parameters:
    -----------
    T_list1, T_list2 : list of numpy.ndarray
        位姿矩阵列表
    angle_unit : str
        角度单位
    
    Returns:
    --------
    dict
        包含统计信息的字典
    """
    
    translation_errors = []
    rotation_errors = []
    
    for T1, T2 in zip(T_list1, T_list2):
        t_err, r_err = calculate_pose_error_simple(T1, T2, angle_unit)
        translation_errors.append(t_err)
        rotation_errors.append(r_err)
    
    translation_errors = np.array(translation_errors)
    rotation_errors = np.array(rotation_errors)
    
    return {
        'translation_errors': translation_errors,
        'rotation_errors': rotation_errors,
        'translation_stats': {
            'mean': np.mean(translation_errors),
            'std': np.std(translation_errors),
            'min': np.min(translation_errors),
            'max': np.max(translation_errors),
            'median': np.median(translation_errors)
        },
        'rotation_stats': {
            'mean': np.mean(rotation_errors),
            'std': np.std(rotation_errors),
            'min': np.min(rotation_errors),
            'max': np.max(rotation_errors),
            'median': np.median(rotation_errors)
        }
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


def apply_transforms_to_points(points, *transforms):
    """
    将多个齐次变换依次应用到点集上
    
    Parameters:
    -----------
    points : numpy.ndarray
        点集，形状为 (N, 3) 或 (3, N) 或 (N, 4) 或 (4, N)
    *transforms : numpy.ndarray
        可变数量的4x4齐次变换矩阵
    
    Returns:
    --------
    numpy.ndarray
        变换后的点集，保持输入的形状格式
    """
    # 统一点的格式为齐次坐标
    original_shape = points.shape
    
    if points.shape[-1] == 3:
        # (N, 3) 格式
        points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
        transpose_needed = False
    elif points.shape[0] == 3:
        # (3, N) 格式
        points_homo = np.vstack([points, np.ones((1, points.shape[1]))])
        transpose_needed = True
    elif points.shape[-1] == 4:
        # (N, 4) 格式
        points_homo = points.copy()
        transpose_needed = False
    elif points.shape[0] == 4:
        # (4, N) 格式
        points_homo = points.copy()
        transpose_needed = True
    else:
        raise ValueError(f"不支持的点集形状: {original_shape}")
    
    # 复合所有变换
    T_total = compose_transforms(*transforms)
    
    # 应用变换
    if transpose_needed:
        # (4, N) 格式
        transformed = T_total @ points_homo
    else:
        # (N, 4) 格式
        transformed = (T_total @ points_homo.T).T
    
    # 恢复原始格式
    if original_shape[-1] == 3:
        return transformed[:, :3]
    elif original_shape[0] == 3:
        return transformed[:3, :]
    else:
        return transformed


def get_intermediate_transforms(transform_list: List[np.ndarray]):
    """
    获取累积的中间变换矩阵
    
    Parameters:
    -----------
    transform_list : List[numpy.ndarray]
        齐次变换矩阵列表
    
    Returns:
    --------
    List[numpy.ndarray]
        中间变换矩阵列表，包含每一步的累积结果
        [T1, T2@T1, T3@T2@T1, ..., Tn@...@T2@T1]
    """
    if len(transform_list) == 0:
        return []
    
    intermediate_transforms = []
    current_transform = transform_list[0].copy()
    intermediate_transforms.append(current_transform.copy())
    
    for T in transform_list[1:]:
        current_transform = current_transform @ T
        intermediate_transforms.append(current_transform.copy())
    
    return intermediate_transforms


def decompose_transform_chain(T_total: np.ndarray, individual_transforms: List[np.ndarray]):
    """
    验证复合变换是否等于总变换（用于调试）
    
    Parameters:
    -----------
    T_total : numpy.ndarray
        期望的总变换矩阵
    individual_transforms : List[numpy.ndarray]
        单个变换矩阵列表
    
    Returns:
    --------
    dict
        包含验证结果的字典
    """
    computed_total = compose_transforms_list(individual_transforms)
    
    # 计算误差
    translation_error = np.linalg.norm(T_total[:3, 3] - computed_total[:3, 3])
    rotation_error = np.linalg.norm(
        Rotation.from_matrix(T_total[:3, :3] @ computed_total[:3, :3].T).as_rotvec()
    )
    
    return {
        'computed_transform': computed_total,
        'translation_error': translation_error,
        'rotation_error_rad': rotation_error,
        'rotation_error_deg': np.degrees(rotation_error),
        'max_element_error': np.max(np.abs(T_total - computed_total)),
        'is_close': np.allclose(T_total, computed_total, rtol=1e-10, atol=1e-10)
    }


def create_transform_from_pose(translation, rotation, rotation_type='euler', degrees=True):
    """
    根据位置和旋转信息创建齐次变换矩阵
    
    Parameters:
    -----------
    translation : array-like
        平移向量 [x, y, z]
    rotation : array-like
        旋转信息，格式取决于rotation_type
    rotation_type : str
        旋转类型：'euler', 'quaternion', 'rotvec', 'matrix'
    degrees : bool
        当rotation_type为'euler'或'rotvec'时，是否使用度制
    
    Returns:
    --------
    numpy.ndarray
        4x4齐次变换矩阵
    """
    T = np.eye(4)
    T[:3, 3] = translation
    
    if rotation_type == 'euler':
        T[:3, :3] = Rotation.from_euler('xyz', rotation, degrees=degrees).as_matrix()
    elif rotation_type == 'quaternion':
        T[:3, :3] = Rotation.from_quat(rotation).as_matrix()
    elif rotation_type == 'rotvec':
        if degrees:
            rotation = np.radians(rotation)
        T[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    elif rotation_type == 'matrix':
        T[:3, :3] = rotation
    else:
        raise ValueError(f"不支持的旋转类型: {rotation_type}")
    
    return T


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


def is_wxyz_format(quaternion):
    """
    检测四元数格式是否为[w,x,y,z]
    基于w分量通常最大的启发式方法
    
    Parameters:
    -----------
    quaternion : array-like
        四元数
    
    Returns:
    --------
    bool
        True if [w,x,y,z] format, False if [x,y,z,w] format
    """
    quaternion = np.array(quaternion)
    # 简单的启发式：如果第一个元素的绝对值最大，可能是w分量
    if np.abs(quaternion[0]) > np.abs(quaternion[3]):
        return True
    return False


def quaternion_multiply(q1, q2):
    """
    四元数乘法
    
    Parameters:
    -----------
    q1, q2 : array-like
        四元数 [x, y, z, w] 格式
    
    Returns:
    --------
    numpy.ndarray
        乘积四元数 [x, y, z, w]
    """
    q1 = np.array(q1)
    q2 = np.array(q2)
    
    # 使用scipy进行四元数乘法
    r1 = Rotation.from_quat(q1)
    r2 = Rotation.from_quat(q2)
    result = r1 * r2
    
    return result.as_quat()


def quaternion_conjugate(quaternion):
    """
    四元数共轭
    
    Parameters:
    -----------
    quaternion : array-like
        四元数 [x, y, z, w]
    
    Returns:
    --------
    numpy.ndarray
        共轭四元数 [-x, -y, -z, w]
    """
    quaternion = np.array(quaternion)
    return np.array([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]])


def quaternion_inverse(quaternion):
    """
    四元数逆
    
    Parameters:
    -----------
    quaternion : array-like
        四元数 [x, y, z, w]
    
    Returns:
    --------
    numpy.ndarray
        逆四元数
    """
    quaternion = normalize_quaternion(quaternion)
    return quaternion_conjugate(quaternion)


def interpolate_poses(pose1, pose2, t):
    """
    在两个位姿之间进行插值
    
    Parameters:
    -----------
    pose1, pose2 : tuple
        位姿 (position, quaternion)，其中position为[x,y,z]，quaternion为[x,y,z,w]
    t : float
        插值参数，0表示pose1，1表示pose2
    
    Returns:
    --------
    tuple
        插值后的位姿 (position, quaternion)
    """
    pos1, quat1 = pose1
    pos2, quat2 = pose2
    
    # 位置线性插值
    pos_interp = np.array(pos1) * (1 - t) + np.array(pos2) * t
    
    # 四元数球面线性插值 (SLERP)
    r1 = Rotation.from_quat(quat1)
    r2 = Rotation.from_quat(quat2)
    
    # 使用scipy的SLERP
    key_rots = Rotation.from_quat([quat1, quat2])
    slerp = key_rots.as_quat()
    
    # 手动SLERP实现
    dot = np.dot(normalize_quaternion(quat1), normalize_quaternion(quat2))
    
    # 如果点积为负，取反一个四元数以选择较短路径
    if dot < 0:
        quat2 = -np.array(quat2)
        dot = -dot
    
    # 线性插值阈值
    if dot > 0.9995:
        # 线性插值
        quat_interp = normalize_quaternion(
            np.array(quat1) * (1 - t) + np.array(quat2) * t
        )
    else:
        # 球面线性插值
        theta = np.arccos(np.abs(dot))
        sin_theta = np.sin(theta)
        
        w1 = np.sin((1 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta
        
        quat_interp = w1 * np.array(quat1) + w2 * np.array(quat2)
    
    return pos_interp, quat_interp


def batch_quaternion_to_homogeneous(quaternions, translations=None):
    """
    批量转换四元数到齐次变换矩阵
    
    Parameters:
    -----------
    quaternions : numpy.ndarray
        形状为 (N, 4) 的四元数数组
    translations : numpy.ndarray, optional
        形状为 (N, 3) 的平移向量数组
    
    Returns:
    --------
    numpy.ndarray
        形状为 (N, 4, 4) 的齐次变换矩阵数组
    """
    quaternions = np.array(quaternions)
    n = quaternions.shape[0]
    
    if translations is None:
        translations = np.zeros((n, 3))
    else:
        translations = np.array(translations)
    
    # 使用scipy批量转换
    rotations = Rotation.from_quat(quaternions)
    rotation_matrices = rotations.as_matrix()
    
    # 构造齐次变换矩阵
    transform_matrices = np.tile(np.eye(4), (n, 1, 1))
    transform_matrices[:, :3, :3] = rotation_matrices
    transform_matrices[:, :3, 3] = translations
    
    return transform_matrices


def batch_homogeneous_to_quaternion(transform_matrices, quaternion_format='xyzw'):
    """
    批量转换齐次变换矩阵到四元数
    
    Parameters:
    -----------
    transform_matrices : numpy.ndarray
        形状为 (N, 4, 4) 的齐次变换矩阵数组
    quaternion_format : str
        输出四元数格式：'xyzw' 或 'wxyz'
    
    Returns:
    --------
    tuple
        (quaternions, translations) 其中：
        - quaternions: 形状为 (N, 4) 的四元数数组
        - translations: 形状为 (N, 3) 的平移向量数组
    """
    transform_matrices = np.array(transform_matrices)
    
    # 提取旋转矩阵和平移向量
    rotation_matrices = transform_matrices[:, :3, :3]
    translations = transform_matrices[:, :3, 3]
    
    # 批量转换为四元数
    rotations = Rotation.from_matrix(rotation_matrices)
    quaternions_xyzw = rotations.as_quat()
    
    if quaternion_format == 'wxyz':
        quaternions = np.column_stack([
            quaternions_xyzw[:, 3],  # w
            quaternions_xyzw[:, 0],  # x
            quaternions_xyzw[:, 1],  # y
            quaternions_xyzw[:, 2]   # z
        ])
    else:
        quaternions = quaternions_xyzw
    
    return quaternions, translations