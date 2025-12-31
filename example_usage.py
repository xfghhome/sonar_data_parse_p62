#!/usr/bin/env python3
"""
示例：如何使用导出的 P62 数据进行自定义分析
"""

import numpy as np
import matplotlib.pyplot as plt

# 加载导出的数据
data = np.load('sonar_data.npz')

timestamps = data['timestamps']
azimuth_indices = data['azimuth_indices']
angles_deg = data['angles_deg']
angles_rad = data['angles_rad']
intensity = data['intensity']
range_max = data['range_max']
num_samples = data['num_samples']
encoder_res = data['encoder_res']

print(f"数据形状: {intensity.shape}")
print(f"方位数: {len(angles_deg)}")
print(f"距离采样点数: {num_samples}")
print(f"最大量程: {range_max} 米")

# 示例1: 计算每个距离单元的平均回波强度
mean_intensity = np.mean(intensity, axis=0)
distance_bins = np.linspace(0, range_max, num_samples)

plt.figure(figsize=(10, 6))
plt.plot(distance_bins, mean_intensity)
plt.xlabel('Distance (m)')
plt.ylabel('Mean Echo Intensity')
plt.title('Mean Echo Intensity vs Distance')
plt.grid(True, alpha=0.3)
plt.savefig('mean_intensity_vs_distance.png', dpi=150)
print("已保存: mean_intensity_vs_distance.png")

# 示例2: 查找最强回波的位置
max_intensity_idx = np.unravel_index(np.argmax(intensity), intensity.shape)
max_azimuth_idx = max_intensity_idx[0]
max_distance_idx = max_intensity_idx[1]

max_angle = angles_deg[max_azimuth_idx]
max_distance = (max_distance_idx / num_samples) * range_max
max_intensity_value = intensity[max_azimuth_idx, max_distance_idx]

print(f"\n最强回波:")
print(f"  角度: {max_angle:.2f}°")
print(f"  距离: {max_distance:.3f} 米")
print(f"  强度: {max_intensity_value}")

# 示例3: 绘制特定角度的回波剖面
target_angle = 90.0  # 90度（右侧）
angle_idx = np.argmin(np.abs(angles_deg - target_angle))
echo_profile = intensity[angle_idx, :]

plt.figure(figsize=(10, 6))
plt.plot(distance_bins, echo_profile)
plt.xlabel('Distance (m)')
plt.ylabel('Echo Intensity')
plt.title(f'Echo Profile at {angles_deg[angle_idx]:.2f}°')
plt.grid(True, alpha=0.3)
plt.savefig('echo_profile_90deg.png', dpi=150)
print(f"已保存: echo_profile_90deg.png")

# 示例4: 统计非零回波的数量
nonzero_count = np.count_nonzero(intensity)
total_pixels = intensity.size
coverage = (nonzero_count / total_pixels) * 100

print(f"\n回波统计:")
print(f"  非零回波数: {nonzero_count} / {total_pixels}")
print(f"  覆盖率: {coverage:.2f}%")

plt.show()

