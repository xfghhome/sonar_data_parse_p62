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
        param3 = struct.unpack('<H', data[18:20])[0]  # 0x12-0x13
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
        angle_deg = (azimuth_idx / encoder_res) * 360.0
        angle_rad = np.deg2rad(angle_deg)
        
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
            'status_byte': status_byte,
            'intensity': intensity,
            'dr': dr,  # 距离分辨率（米）
        }
        
        return record
    
    def get_statistics(self) -> Dict:
        """获取文件统计信息"""
        if not self.records:
            return {}
        
        timestamps = [r['timestamp'] for r in self.records]
        azimuths = [r['azimuth_idx'] for r in self.records]
        angles = [r['angle_deg'] for r in self.records]
        
        stats = {
            'num_records': len(self.records),
            'time_span_sec': (max(timestamps) - min(timestamps)) if len(timestamps) > 1 else 0,
            'time_start': min(timestamps),
            'time_end': max(timestamps),
            'azimuth_min': min(azimuths),
            'azimuth_max': max(azimuths),
            'angle_min': min(angles),
            'angle_max': max(angles),
            'range_max': self.records[0]['range_max'] if self.records else 0,
            'num_samples': self.records[0]['param2'] if self.records else 0,
        }
        
        return stats


def visualize_polar(parser: P62Parser, output_file: str = 'polar_view.png'):
    """生成极坐标可视化"""
    records = parser.records
    if not records:
        print("没有可用的记录")
        return
    
    # 创建极坐标图像
    num_samples = records[0]['param2']  # 1348
    range_max = records[0]['range_max']  # 5.0 m
    
    # 按角度排序
    sorted_records = sorted(records, key=lambda x: x['azimuth_idx'])
    
    # 构建极坐标矩阵
    # 行：角度（方位）
    # 列：距离
    image = np.zeros((len(sorted_records), num_samples))
    
    for i, record in enumerate(sorted_records):
        image[i, :] = record['intensity']
    
    # 绘制
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # 创建角度和距离数组
    angles = np.array([r['angle_rad'] for r in sorted_records])
    ranges = np.linspace(0, range_max, num_samples)
    
    # 创建网格 - 扩展网格以匹配 pcolormesh 的要求
    # 对于 shading='flat'，网格维度应该比数据多1
    # 对于 shading='auto'，可以使用相同维度，但扩展更安全
    if len(angles) > 1:
        angle_step = angles[1] - angles[0]
        angles_extended = np.append(angles, angles[-1] + angle_step)
    else:
        angles_extended = np.append(angles, angles[-1] + 0.01)
    
    if len(ranges) > 1:
        range_step = ranges[1] - ranges[0]
        ranges_extended = np.append(ranges, ranges[-1] + range_step)
    else:
        ranges_extended = np.append(ranges, ranges[-1] + 0.01)
    
    # 使用 'ij' 索引：第一个参数对应行（角度），第二个参数对应列（距离）
    theta_grid, r_grid = np.meshgrid(angles_extended, ranges_extended, indexing='ij')
    
    # 绘制极坐标图像
    # image 的形状是 (角度数, 距离数) = (1506, 1348)
    # 网格现在是 (1507, 1349)，与 shading='flat' 兼容
    try:
        im = ax.pcolormesh(theta_grid, r_grid, image, cmap='hot', shading='flat', vmin=0, vmax=15)
    except TypeError:
        # 如果 shading='flat' 失败，尝试 shading='auto'
        im = ax.pcolormesh(theta_grid, r_grid, image, cmap='hot', shading='auto', vmin=0, vmax=15)
    ax.set_theta_zero_location('N')  # 0度在顶部（北）
    ax.set_theta_direction(-1)  # 顺时针方向（可根据需要调整）
    ax.set_ylim(0, range_max)
    ax.set_title('Polar View - Omnidirectional Sonar', fontsize=14, pad=20)
    
    plt.colorbar(im, ax=ax, label='Echo Intensity (4-bit)')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"极坐标图像已保存: {output_file}")
    plt.close()


def visualize_cartesian(parser: P62Parser, output_file: str = 'cartesian_view.png'):
    """生成笛卡尔坐标可视化"""
    records = parser.records
    if not records:
        print("没有可用的记录")
        return
    
    # 按角度排序
    sorted_records = sorted(records, key=lambda x: x['azimuth_idx'])
    
    num_samples = records[0]['param2']  # 1348
    range_max = records[0]['range_max']  # 5.0 m
    
    # 构建笛卡尔坐标图像 - 使用更高效的方法
    img_size = 1000
    image = np.zeros((img_size, img_size), dtype=np.float32)
    count = np.zeros((img_size, img_size), dtype=np.int32)  # 用于平均
    
    center = img_size // 2
    scale = (img_size // 2) / range_max
    
    # 批量处理所有记录
    for record in sorted_records:
        angle = record['angle_rad']
        intensity = record['intensity']
        
        # 计算距离数组
        distances = np.arange(num_samples) * (range_max / num_samples)
        
        # 转换为笛卡尔坐标
        x_coords = center + distances * scale * np.cos(angle)
        y_coords = center + distances * scale * np.sin(angle)
        
        # 转换为整数像素坐标
        x_pixels = np.clip(x_coords.astype(int), 0, img_size - 1)
        y_pixels = np.clip(y_coords.astype(int), 0, img_size - 1)
        
        # 累加强度值（使用最大值而不是平均值，效果更好）
        for i in range(num_samples):
            px, py = x_pixels[i], y_pixels[i]
            if intensity[i] > 0:
                image[py, px] = max(image[py, px], intensity[i])
    
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


def export_data(parser: P62Parser, output_file: str = 'sonar_data.npz'):
    """导出解析后的数据为 NumPy 格式"""
    records = parser.records
    if not records:
        print("没有可用的记录")
        return
    
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
    print(f"时间跨度: {stats['time_span_sec']:.2f} 秒")
    print(f"起始时间: {stats['time_start']} ({records[0]['time_str']})")
    print(f"结束时间: {stats['time_end']} ({records[-1]['time_str']})")
    print(f"方位索引范围: {stats['azimuth_min']} - {stats['azimuth_max']}")
    print(f"角度范围: {stats['angle_min']:.2f}° - {stats['angle_max']:.2f}°")
    print(f"最大量程: {stats['range_max']:.2f} 米")
    print(f"距离采样点数: {stats['num_samples']}")
    
    # 生成可视化
    if args.all or args.polar:
        visualize_polar(p62_parser)
    
    if args.all or args.cartesian:
        visualize_cartesian(p62_parser)
    
    if args.all or args.scan:
        visualize_scan_pattern(p62_parser)
    
    if args.export:
        export_data(p62_parser, args.export)
    
    if not (args.all or args.polar or args.cartesian or args.scan or args.export):
        print("\n提示: 使用 --all 生成所有可视化，或使用 --polar, --cartesian, --scan 选择特定视图")
        print("     使用 --export <filename.npz> 导出数据为 NumPy 格式")


if __name__ == '__main__':
    main()

