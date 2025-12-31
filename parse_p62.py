#!/usr/bin/env python3
"""
P62 文件解析器 - 全向声呐数据解析与可视化
根据文件结构分析，解析每条记录并生成极坐标/笛卡尔坐标可视化
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict
import argparse
import matplotlib

# 配置中文字体支持
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    pass


class P62Parser:
    """P62 文件解析器"""
    
    RECORD_SIZE = 695
    HEADER_SIZE = 21
    MAGIC = b'\xFE\x14\x10\x01'
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.records = []
        
    def parse(self) -> List[Dict]:
        """解析整个文件，返回所有记录"""
        with open(self.filepath, 'rb') as f:
            data = f.read()
        
        file_size = len(data)
        num_records = file_size // self.RECORD_SIZE
        
        print(f"文件大小: {file_size} 字节")
        print(f"记录数: {num_records}")
        print(f"每条记录: {self.RECORD_SIZE} 字节")
        
        records = []
        for i in range(num_records):
            offset = i * self.RECORD_SIZE
            record_data = data[offset:offset + self.RECORD_SIZE]
            
            if len(record_data) < self.RECORD_SIZE:
                break
                
            record = self._parse_record(record_data, i)
            if record:
                records.append(record)
        
        self.records = records
        print(f"成功解析 {len(records)} 条记录")
        return records
    
    def _parse_record(self, data: bytes, index: int) -> Dict:
        """解析单条记录"""
        if len(data) < self.RECORD_SIZE:
            return None
        
        # 验证 magic
        magic = data[0:4]
        if magic != self.MAGIC:
            print(f"警告: 记录 {index} magic 不匹配: {magic.hex()}")
        
        # 解析头部字段（小端序）
        timestamp = struct.unpack('<I', data[4:8])[0]  # 0x04-0x07
        azimuth_idx = struct.unpack('<H', data[8:10])[0]  # 0x08-0x09
        param1 = struct.unpack('<H', data[10:12])[0]  # 0x0A-0x0B (1313)
        param2 = struct.unpack('<H', data[12:14])[0]  # 0x0C-0x0D (1348)
        range_setting = struct.unpack('<H', data[14:16])[0]  # 0x0E-0x0F (50)
        encoder_res = struct.unpack('<H', data[16:18])[0]  # 0x10-0x11 (1568)
        # param3: 同时解析两种端序以验证
        param3_raw = data[18:20]
        param3_le = struct.unpack('<H', param3_raw)[0]  # 小端
        param3_be = struct.unpack('>H', param3_raw)[0]  # 大端
        param3 = param3_le  # 默认使用小端，但保留大端值用于统计
        status_byte = data[20]  # 0x14
        
        # 解析回波数据（从0x15开始，674字节）
        echo_data = data[21:21+674]
        
        # 解包4-bit数据：每个字节包含两个4-bit样本
        # 低4位在前，高4位在后
        intensity = []
        for byte in echo_data:
            lo = byte & 0x0F
            hi = (byte >> 4) & 0x0F
            intensity.append(lo)
            intensity.append(hi)
        
        # 只取前1348个（因为674字节 = 1348个4-bit样本）
        intensity = np.array(intensity[:1348], dtype=np.float32)
        
        # 计算时间（Unix时间戳）
        from datetime import datetime, timezone
        try:
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            time_str = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        except:
            time_str = f"Timestamp: {timestamp}"
        
        # 计算角度（假设0-1568对应0-360度）
        # 注意：这里先计算设备角度，后续在可视化时会转换为统一的绘图角度
        angle_deg_device = (azimuth_idx / encoder_res) * 360.0
        angle_rad_device = np.deg2rad(angle_deg_device)
        
        # 统一角度约定：转换为"北为0°、顺时针为正"的绘图角度
        # 设备角度通常是"东为0°、逆时针为正"，需要转换
        # theta_plot = (π/2) - theta_device  (转换为北为0°)
        # 但考虑到极坐标图使用 set_theta_direction(-1)，我们保持设备角度
        # 在可视化时统一处理
        angle_deg = angle_deg_device
        angle_rad = angle_rad_device
        
        # 计算距离分辨率
        range_max = range_setting * 0.1  # 假设单位是0.1m
        dr = range_max / param2  # 距离分辨率
        
        record = {
            'index': index,
            'magic': magic.hex(),
            'timestamp': timestamp,
            'time_str': time_str,
            'azimuth_idx': azimuth_idx,
            'angle_deg': angle_deg,
            'angle_rad': angle_rad,
            'param1': param1,
            'param2': param2,  # 1348 - 采样点数
            'range_setting': range_setting,
            'range_max': range_max,  # 最大量程（米）
            'encoder_res': encoder_res,  # 1568
            'param3': param3,
            'param3_le': param3_le,
            'param3_be': param3_be,
            'param3_raw': param3_raw.hex(),
            'status_byte': status_byte,
            'intensity': intensity,
            'dr': dr,  # 距离分辨率（米）
        }
        
        return record
    
    def detect_scan_cycles(self, records: List[Dict]) -> List[List[Dict]]:
        """
        检测扫描圈数，将记录按圈分组
        当azimuth_idx从大突然变小（回落），认为进入下一圈
        """
        if not records:
            return []
        
        cycles = []
        current_cycle = [records[0]]
        
        for i in range(1, len(records)):
            prev_azimuth = records[i-1]['azimuth_idx']
            curr_azimuth = records[i]['azimuth_idx']
            
            # 检测回落：如果当前azimuth比前一个小很多（超过阈值），认为是新的一圈
            # 阈值设为encoder_res的一半，避免小跳变误判
            threshold = records[0].get('encoder_res', 1568) // 2
            if curr_azimuth < prev_azimuth - threshold:
                # 开始新的一圈
                if current_cycle:
                    cycles.append(current_cycle)
                current_cycle = [records[i]]
            else:
                current_cycle.append(records[i])
        
        if current_cycle:
            cycles.append(current_cycle)
        
        return cycles
    
    def extract_single_cycle(self, records: List[Dict], cycle_idx: int = 0, aggregate: bool = True, 
                            apply_azimuth_offset: bool = True) -> List[Dict]:
        """
        提取单圈扫描数据
        cycle_idx: 要提取的圈索引（0为第一圈）
        aggregate: 如果True，对相同azimuth_idx的记录取最大值；如果False，只取第一圈
        apply_azimuth_offset: 如果True，将每圈的第一个azimuth_idx作为0°基准
        """
        cycles = self.detect_scan_cycles(records)
        
        if not cycles:
            return records
        
        if cycle_idx >= len(cycles):
            cycle_idx = 0
            print(f"警告: 请求的圈索引 {cycle_idx} 超出范围，使用第一圈")
        
        selected_cycle = cycles[cycle_idx]
        
        # 应用零位偏置：将第一条的azimuth_idx作为0°基准
        if apply_azimuth_offset and selected_cycle:
            azimuth_offset = selected_cycle[0]['azimuth_idx']
            encoder_res = selected_cycle[0].get('encoder_res', 1568)
            print(f"应用零位偏置: azimuth_offset={azimuth_offset}")
            
            # 更新记录的角度（相对偏移）
            for record in selected_cycle:
                # 计算相对azimuth_idx（取模处理）
                relative_azimuth = (record['azimuth_idx'] - azimuth_offset) % encoder_res
                # 重新计算角度
                angle_deg = (relative_azimuth / encoder_res) * 360.0
                angle_rad = np.deg2rad(angle_deg)
                record['azimuth_idx_original'] = record['azimuth_idx']
                record['azimuth_idx'] = relative_azimuth
                record['angle_deg'] = angle_deg
                record['angle_rad'] = angle_rad
        
        if aggregate:
            # 对相同azimuth_idx的记录聚合（取最大值）
            from collections import defaultdict
            azimuth_dict = defaultdict(list)
            for record in selected_cycle:
                azimuth_dict[record['azimuth_idx']].append(record)
            
            aggregated = []
            for azimuth_idx in sorted(azimuth_dict.keys()):
                records_same_azimuth = azimuth_dict[azimuth_idx]
                # 取强度最大的记录
                best_record = max(records_same_azimuth, key=lambda r: r['intensity'].max())
                aggregated.append(best_record)
            
            return aggregated
        else:
            return selected_cycle
    
    def get_statistics(self) -> Dict:
        """获取文件统计信息"""
        if not self.records:
            return {}
        
        timestamps = [r['timestamp'] for r in self.records]
        azimuths = [r['azimuth_idx'] for r in self.records]
        angles = [r['angle_deg'] for r in self.records]
        
        # 统计param3的两种端序
        param3_le_values = [r.get('param3_le', r.get('param3', 0)) for r in self.records]
        param3_be_values = [r.get('param3_be', r.get('param3', 0)) for r in self.records]
        
        # 检测圈数
        cycles = self.detect_scan_cycles(self.records)
        
        stats = {
            'num_records': len(self.records),
            'num_cycles': len(cycles),
            'time_span_sec': (max(timestamps) - min(timestamps)) if len(timestamps) > 1 else 0,
            'time_start': min(timestamps),
            'time_end': max(timestamps),
            'azimuth_min': min(azimuths),
            'azimuth_max': max(azimuths),
            'angle_min': min(angles),
            'angle_max': max(angles),
            'range_max': self.records[0]['range_max'] if self.records else 0,
            'num_samples': self.records[0]['param2'] if self.records else 0,
            'param1': self.records[0]['param1'] if self.records else 0,
            'param2': self.records[0]['param2'] if self.records else 0,
            'param3_le_common': max(set(param3_le_values), key=param3_le_values.count) if param3_le_values else 0,
            'param3_be_common': max(set(param3_be_values), key=param3_be_values.count) if param3_be_values else 0,
            'param3_le_range': (min(param3_le_values), max(param3_le_values)) if param3_le_values else (0, 0),
            'param3_be_range': (min(param3_be_values), max(param3_be_values)) if param3_be_values else (0, 0),
        }
        
        return stats


def convert_angle_for_plotting(angle_rad_device: np.ndarray, use_polar_convention: bool = True) -> np.ndarray:
    """
    统一角度约定转换
    use_polar_convention: 如果True，转换为"北为0°、顺时针为正"（用于极坐标图）
                         如果False，转换为标准数学约定"东为0°、逆时针为正"（用于笛卡尔图）
    为了统一，我们使用标准数学约定，然后在极坐标图中通过set_theta_zero_location和set_theta_direction调整
    """
    # 标准数学约定：东为0°，逆时针为正
    # 设备角度通常也是这个约定，所以直接返回
    # 如果需要"北为0°、顺时针"，则：theta_plot = (π/2) - theta_device
    if use_polar_convention:
        # 转换为"北为0°、顺时针为正"
        return (np.pi / 2) - angle_rad_device
    else:
        # 标准数学约定
        return angle_rad_device


def visualize_polar(parser: P62Parser, output_file: str = 'polar_view.png', 
                    use_single_cycle: bool = True, cycle_idx: int = 0,
                    use_valid_range_only: bool = False, apply_azimuth_offset: bool = True):
    """生成极坐标可视化"""
    records = parser.records
    if not records:
        print("没有可用的记录")
        return
    
    # 切圈处理：只使用单圈数据
    if use_single_cycle:
        records = parser.extract_single_cycle(records, cycle_idx=cycle_idx, aggregate=True,
                                             apply_azimuth_offset=apply_azimuth_offset)
        print(f"使用单圈数据: {len(records)} 条记录")
    
    if not records:
        print("没有可用的记录")
        return
    
    # 按角度排序
    sorted_records = sorted(records, key=lambda x: x['azimuth_idx'])
    
    # 获取参数
    param1 = records[0]['param1']  # 有效距离bin数 (1313)
    param2 = records[0]['param2']  # 总采样点数 (1348)
    range_max = records[0]['range_max']  # 5.0 m
    
    # 确定使用的采样点数
    if use_valid_range_only:
        num_samples = param1
        skip_bins = param2 - param1
        print(f"只使用有效范围: {num_samples} 个采样点 (跳过前/后 {skip_bins} 个bin)")
    else:
        num_samples = param2
    
    # 构建极坐标矩阵
    # 行：角度（方位）
    # 列：距离
    image = np.zeros((len(sorted_records), num_samples))
    
    for i, record in enumerate(sorted_records):
        intensity = record['intensity']
        if use_valid_range_only:
            # 只取有效部分（假设是中间部分，或跳过前skip_bins/2和后skip_bins/2）
            skip_bins = param2 - param1
            skip_start = skip_bins // 2
            intensity = intensity[skip_start:skip_start + param1]
        image[i, :] = intensity[:num_samples]
    
    # 绘制
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # 统一角度约定：转换为绘图角度
    angles_device = np.array([r['angle_rad'] for r in sorted_records])
    # 对角度做unwrap，处理跨0°的情况
    angles_unwrapped = np.unwrap(angles_device)
    # 转换为极坐标图约定（北为0°、顺时针为正）
    angles_plot = convert_angle_for_plotting(angles_unwrapped, use_polar_convention=True)
    
    # 创建距离数组：使用bin center和bin edge
    # bin center
    r_centers = (np.arange(num_samples) + 0.5) * (range_max / num_samples)
    # bin edges (比centers多1个)
    r_edges = np.linspace(0, range_max, num_samples + 1)
    
    # 创建角度边界：使用mid-point方式
    if len(angles_plot) > 1:
        # 计算相邻角度的中点作为边界
        angles_mid = (angles_plot[:-1] + angles_plot[1:]) / 2
        # 第一个边界：第一个角度减去到中点的距离
        first_edge = angles_plot[0] - (angles_mid[0] - angles_plot[0])
        # 最后一个边界：最后一个角度加上到中点的距离
        last_edge = angles_plot[-1] + (angles_plot[-1] - angles_mid[-1])
        theta_edges = np.concatenate([[first_edge], angles_mid, [last_edge]])
    else:
        # 如果只有一条记录，使用一个小的角度范围
        theta_edges = np.array([angles_plot[0] - 0.01, angles_plot[0] + 0.01])
    
    # 使用 'ij' 索引：第一个参数对应行（角度），第二个参数对应列（距离）
    theta_grid, r_grid = np.meshgrid(theta_edges, r_edges, indexing='ij')
    
    # 绘制极坐标图像
    try:
        im = ax.pcolormesh(theta_grid, r_grid, image, cmap='hot', shading='flat', vmin=0, vmax=15)
    except (TypeError, ValueError) as e:
        # 如果 shading='flat' 失败，尝试 shading='auto'
        print(f"警告: pcolormesh使用flat模式失败: {e}, 改用auto模式")
        # 对于auto模式，使用centers
        theta_centers_grid, r_centers_grid = np.meshgrid(angles_plot, r_centers, indexing='ij')
        im = ax.pcolormesh(theta_centers_grid, r_centers_grid, image, cmap='hot', shading='auto', vmin=0, vmax=15)
    
    ax.set_theta_zero_location('N')  # 0度在顶部（北）
    ax.set_theta_direction(-1)  # 顺时针方向
    ax.set_ylim(0, range_max)
    ax.set_title('Polar View - Omnidirectional Sonar', fontsize=14, pad=20)
    
    plt.colorbar(im, ax=ax, label='Echo Intensity (4-bit)')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"极坐标图像已保存: {output_file}")
    plt.close()


def visualize_cartesian(parser: P62Parser, output_file: str = 'cartesian_view.png',
                        use_single_cycle: bool = True, cycle_idx: int = 0,
                        use_valid_range_only: bool = False, apply_azimuth_offset: bool = True):
    """生成笛卡尔坐标可视化"""
    records = parser.records
    if not records:
        print("没有可用的记录")
        return
    
    # 切圈处理：只使用单圈数据
    if use_single_cycle:
        records = parser.extract_single_cycle(records, cycle_idx=cycle_idx, aggregate=True,
                                             apply_azimuth_offset=apply_azimuth_offset)
        print(f"使用单圈数据: {len(records)} 条记录")
    
    if not records:
        print("没有可用的记录")
        return
    
    # 按角度排序
    sorted_records = sorted(records, key=lambda x: x['azimuth_idx'])
    
    param1 = records[0]['param1']  # 有效距离bin数 (1313)
    param2 = records[0]['param2']  # 总采样点数 (1348)
    range_max = records[0]['range_max']  # 5.0 m
    
    # 确定使用的采样点数
    if use_valid_range_only:
        num_samples = param1
        skip_bins = param2 - param1
        skip_start = skip_bins // 2
        print(f"只使用有效范围: {num_samples} 个采样点 (跳过前 {skip_start} 个bin)")
    else:
        num_samples = param2
        skip_start = 0
    
    # 构建笛卡尔坐标图像 - 使用向量化方法
    img_size = 1000
    image = np.zeros((img_size, img_size), dtype=np.float32)
    
    center = img_size // 2
    scale = (img_size // 2) / range_max
    
    # 统一角度约定：转换为标准数学约定（东为0°、逆时针为正）
    # 与极坐标图保持一致，使用相同的角度转换
    angles_device = np.array([r['angle_rad'] for r in sorted_records])
    angles_unwrapped = np.unwrap(angles_device)
    # 笛卡尔图使用标准数学约定
    angles_plot = convert_angle_for_plotting(angles_unwrapped, use_polar_convention=False)
    
    # 批量处理所有记录 - 向量化
    for i, record in enumerate(sorted_records):
        intensity = record['intensity']
        if use_valid_range_only:
            intensity = intensity[skip_start:skip_start + param1]
        else:
            intensity = intensity[:num_samples]
        
        # 计算距离数组（bin centers）
        distances = (np.arange(len(intensity)) + 0.5) * (range_max / num_samples)
        
        # 使用统一的角度约定
        angle = angles_plot[i]
        
        # 转换为笛卡尔坐标（标准数学约定：x=r*cos(θ), y=r*sin(θ)）
        # 但为了与极坐标图一致（北为0°），我们需要调整
        # 如果极坐标图是"北为0°、顺时针"，那么笛卡尔图应该也是
        # 所以使用：x = r*sin(θ), y = r*cos(θ) 来匹配
        # 或者统一使用极坐标图的角度转换
        angle_for_cart = convert_angle_for_plotting(np.array([angle]), use_polar_convention=True)[0]
        x_coords = center + distances * scale * np.sin(angle_for_cart)  # 注意：sin对应x
        y_coords = center + distances * scale * np.cos(angle_for_cart)  # cos对应y
        
        # 转换为整数像素坐标
        x_pixels = np.clip(x_coords.astype(int), 0, img_size - 1)
        y_pixels = np.clip(y_coords.astype(int), 0, img_size - 1)
        
        # 向量化累加：使用np.maximum.at（只处理有效强度）
        valid = intensity > 0
        if np.any(valid):
            np.maximum.at(image, (y_pixels[valid], x_pixels[valid]), intensity[valid])
    
    # 绘制
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(image, cmap='hot', origin='lower', extent=[-range_max, range_max, -range_max, range_max])
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Cartesian View - Omnidirectional Sonar', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.colorbar(im, ax=ax, label='Echo Intensity (4-bit)')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"笛卡尔坐标图像已保存: {output_file}")
    plt.close()


def visualize_scan_pattern(parser: P62Parser, output_file: str = 'scan_pattern.png'):
    """可视化扫描模式（方位索引随时间变化）"""
    records = parser.records
    if not records:
        return
    
    indices = [r['index'] for r in records]
    azimuths = [r['azimuth_idx'] for r in records]
    timestamps = [r['timestamp'] for r in records]
    
    # 相对时间（秒）
    time_rel = [(t - timestamps[0]) for t in timestamps]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 方位索引随时间变化
    ax1.plot(time_rel, azimuths, 'b.', markersize=2)
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Azimuth Index', fontsize=12)
    ax1.set_title('Azimuth Index vs Time', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 方位索引分布直方图
    ax2.hist(azimuths, bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Azimuth Index', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Azimuth Index Distribution', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"扫描模式图已保存: {output_file}")
    plt.close()


def export_data(parser: P62Parser, output_file: str = 'sonar_data.npz', 
                use_single_cycle: bool = True, cycle_idx: int = 0, apply_azimuth_offset: bool = True):
    """导出解析后的数据为 NumPy 格式"""
    records = parser.records
    if not records:
        print("没有可用的记录")
        return
    
    # 切圈处理
    if use_single_cycle:
        records = parser.extract_single_cycle(records, cycle_idx=cycle_idx, aggregate=True,
                                             apply_azimuth_offset=apply_azimuth_offset)
        print(f"导出单圈数据: {len(records)} 条记录")
    
    # 按角度排序
    sorted_records = sorted(records, key=lambda x: x['azimuth_idx'])
    
    # 提取数据
    timestamps = np.array([r['timestamp'] for r in sorted_records])
    azimuth_indices = np.array([r['azimuth_idx'] for r in sorted_records])
    angles_deg = np.array([r['angle_deg'] for r in sorted_records])
    angles_rad = np.array([r['angle_rad'] for r in sorted_records])
    
    # 构建强度矩阵
    num_samples = records[0]['param2']
    intensity_matrix = np.array([r['intensity'] for r in sorted_records])
    
    # 保存
    np.savez_compressed(
        output_file,
        timestamps=timestamps,
        azimuth_indices=azimuth_indices,
        angles_deg=angles_deg,
        angles_rad=angles_rad,
        intensity=intensity_matrix,
        range_max=records[0]['range_max'],
        num_samples=num_samples,
        encoder_res=records[0]['encoder_res']
    )
    
    print(f"数据已导出: {output_file}")
    print(f"  时间戳数组: {len(timestamps)} 个")
    print(f"  强度矩阵: {intensity_matrix.shape} (方位数 x 距离采样点数)")


def main():
    parser = argparse.ArgumentParser(description='P62 文件解析与可视化')
    parser.add_argument('input_file', type=str, help='输入的P62文件路径')
    parser.add_argument('--polar', action='store_true', help='生成极坐标视图')
    parser.add_argument('--cartesian', action='store_true', help='生成笛卡尔坐标视图')
    parser.add_argument('--scan', action='store_true', help='生成扫描模式图')
    parser.add_argument('--all', action='store_true', help='生成所有可视化')
    parser.add_argument('--export', type=str, help='导出数据为NPZ格式（指定输出文件名）')
    parser.add_argument('--single-cycle', action='store_true', default=True, 
                       dest='single_cycle', help='只使用单圈扫描数据（默认开启）')
    parser.add_argument('--all-cycles', action='store_false', dest='single_cycle',
                       help='使用所有圈的数据')
    parser.add_argument('--cycle-idx', type=int, default=0,
                       help='要使用的圈索引（默认0，第一圈）')
    parser.add_argument('--valid-range-only', action='store_true',
                       help='只使用param1指示的有效距离范围')
    
    args = parser.parse_args()
    
    # 解析文件
    p62_parser = P62Parser(args.input_file)
    records = p62_parser.parse()
    
    if not records:
        print("错误: 未能解析任何记录")
        return
    
    # 显示统计信息
    stats = p62_parser.get_statistics()
    print("\n=== 文件统计信息 ===")
    print(f"记录数: {stats['num_records']}")
    print(f"扫描圈数: {stats['num_cycles']}")
    print(f"时间跨度: {stats['time_span_sec']:.2f} 秒")
    print(f"起始时间: {stats['time_start']} ({records[0]['time_str']})")
    print(f"结束时间: {stats['time_end']} ({records[-1]['time_str']})")
    print(f"方位索引范围: {stats['azimuth_min']} - {stats['azimuth_max']}")
    print(f"角度范围: {stats['angle_min']:.2f}° - {stats['angle_max']:.2f}°")
    print(f"最大量程: {stats['range_max']:.2f} 米")
    print(f"距离采样点数: {stats['num_samples']} (param2)")
    print(f"有效距离bin数: {stats['param1']} (param1)")
    print(f"param1/param2差异: {stats['param2'] - stats['param1']} 个bin")
    print(f"\nparam3 端序分析:")
    print(f"  小端 (LE) 常见值: {stats['param3_le_common']} (范围: {stats['param3_le_range'][0]}-{stats['param3_le_range'][1]})")
    print(f"  大端 (BE) 常见值: {stats['param3_be_common']} (范围: {stats['param3_be_range'][0]}-{stats['param3_be_range'][1]})")
    
    # 生成可视化
    if args.all or args.polar:
        visualize_polar(p62_parser, use_single_cycle=args.single_cycle, 
                       cycle_idx=args.cycle_idx, use_valid_range_only=args.valid_range_only,
                       apply_azimuth_offset=True)
    
    if args.all or args.cartesian:
        visualize_cartesian(p62_parser, use_single_cycle=args.single_cycle,
                           cycle_idx=args.cycle_idx, use_valid_range_only=args.valid_range_only,
                           apply_azimuth_offset=True)
    
    if args.all or args.scan:
        visualize_scan_pattern(p62_parser)
    
    if args.export:
        export_data(p62_parser, args.export, use_single_cycle=args.single_cycle, 
                   cycle_idx=args.cycle_idx)
    
    if not (args.all or args.polar or args.cartesian or args.scan or args.export):
        print("\n提示: 使用 --all 生成所有可视化，或使用 --polar, --cartesian, --scan 选择特定视图")
        print("     使用 --export <filename.npz> 导出数据为 NumPy 格式")


if __name__ == '__main__':
    main()

