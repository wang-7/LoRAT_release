import numpy as np
from scipy.io import loadmat
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

# 加载数据
file_name = '/mnt/data/WQ/Raw-CSI-Data/ArgosCSI-96x8-2016-11-04-05-37-37_2.4GHz_track_left_to_right_NLOS.npy'
file_name2 = '/mnt/data/WQ/Raw-CSI-Data/ArgosCSI-96x8-2016-05-01-06-57-58-2.4GHz-continuousmobile.npy'
file_name3 = '/mnt/data/WQ/Raw-CSI-Data/ArgosCSI-96x2-2016-12-07-03-00-36_rotation_mob_horizontal_omni.npy'
file_name4 = '/mnt/data/WQ/Argos/train_18_test_6_antenna_96_subcarriers_52/train/downlink/ArgosCSI-96x8-2016-05-01-07-17-44-5GHz-continuousmobility.mat'
file_name5 = '/mnt/data/WQ/Argos/train_18_test_6_antenna_96_subcarriers_52/train/downlink/ArgosCSI-96x2-2016-12-07-03-53-37_rotation_linear_mob_patch.mat'
file_name6 = '/mnt/data/WQ/Argos/train_18_test_6_antenna_96_subcarriers_52/train/downlink/ArgosCSI-96x2-2016-03-31-15-53-27_Jian_left_to_right.mat'

ant_idx = 2
data = np.load(file_name)[:5026, 0, ant_idx, :]
data2 = np.load(file_name2)[:5026, 0, ant_idx, :]
data3 = np.load(file_name3)[:5026, 0, ant_idx, :]
data4 = loadmat(file_name4)['data'][:5026, 0, ant_idx, :]
data5 = loadmat(file_name5)['data'][:5026, 0, ant_idx, :]
data6 = loadmat(file_name6)['data'][:5026, 0, ant_idx, :]

# 定义known_env和unknown_env
known_env = [data, data2, data3, data5, data6]
unknown_env = [data4]

# 计算KL散度的函数
def calculate_kl_divergence(p, q, epsilon=1e-10):
    """
    计算两个概率分布之间的KL散度
    """
    # 添加小的epsilon值以避免除零错误
    p = p + epsilon
    q = q + epsilon
    
    # 归一化为概率分布
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # 计算KL散度
    return entropy(p, q)

# 将复数数据转换为幅度
def complex_to_magnitude(data):
    """
    将复数数据转换为幅度
    """
    return np.abs(data)

# 计算所有known_env数据对之间的KL散度
kl_divergences = []

print("计算known_env中两两之间数据的KL散度...")

# 将复数数据转换为幅度并展平
magnitude_data = []
for i, env_data in enumerate(known_env):
    mag_data = complex_to_magnitude(env_data)
    # 展平数据以便计算KL散度
    flattened_data = mag_data.flatten()
    magnitude_data.append(flattened_data)
    print(f"data{i+1} shape: {flattened_data.shape}")

# 计算两两之间的KL散度
for i in range(len(magnitude_data)):
    for j in range(i+1, len(magnitude_data)):
        # 确保两个分布具有相同的长度
        min_len = min(len(magnitude_data[i]), len(magnitude_data[j]))
        p = magnitude_data[i][:min_len]
        q = magnitude_data[j][:min_len]
        
        # 计算KL散度 (D_KL(P||Q) 和 D_KL(Q||P))
        kl_pq = calculate_kl_divergence(p, q)
        kl_qp = calculate_kl_divergence(q, p)
        
        # 计算对称KL散度
        sym_kl = (kl_pq + kl_qp) / 2
        
        # 计算JS散度作为另一种度量
        js_div = jensenshannon(p/np.sum(p), q/np.sum(q))**2
        
        kl_divergences.append({
            'pair': (i+1, j+1),
            'KL(P||Q)': kl_pq,
            'KL(Q||P)': kl_qp,
            'Symmetric KL': sym_kl,
            'JS divergence': js_div
        })
        
        print(f"KL散度 between data{i+1} and data{j+1}:")
        print(f"  D_KL(P||Q) = {kl_pq:.6f}")
        print(f"  D_KL(Q||P) = {kl_qp:.6f}")
        print(f"  Symmetric KL = {sym_kl:.6f}")
        print(f"  JS divergence = {js_div:.6f}")
        print()

# 保存结果
import json
with open('/mnt/data/WQ/LoRAT/kl_divergence_results.json', 'w') as f:
    json.dump(kl_divergences, f, indent=2, default=str)

print("KL散度计算完成，结果已保存到 /mnt/data/WQ/LoRAT/kl_divergence_results.json")