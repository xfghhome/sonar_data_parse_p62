#!/usr/bin/env python3
"""
P62 文件解析器 - 简化版（不依赖matplotlib，用于验证解析逻辑）
"""

import struct
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timezone


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
        intensity = intensity[:1348]
        
        # 计算时间（Unix时间戳）
        try:
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            time_str = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        except:
            time_str = f"Timestamp: {timestamp}"
        
        # 计算角度（假设0-1568对应0-360度）
        angle_deg = (azimuth_idx / encoder_res) * 360.0
        
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
    
    def print_sample_records(self, num: int = 5):
        """打印前几条记录的信息"""
        if not self.records:
            print("没有记录")
            return
        
        print(f"\n=== 前 {min(num, len(self.records))} 条记录详情 ===")
        for i, record in enumerate(self.records[:num]):
            print(f"\n记录 {i}:")
            print(f"  Magic: {record['magic']}")
            print(f"  时间戳: {record['timestamp']} ({record['time_str']})")
            print(f"  方位索引: {record['azimuth_idx']}")
            print(f"  角度: {record['angle_deg']:.2f}°")
            print(f"  量程设置: {record['range_setting']} (最大量程: {record['range_max']:.2f} m)")
            print(f"  采样点数: {record['param2']}")
            print(f"  编码器分辨率: {record['encoder_res']}")
            print(f"  回波强度范围: {min(record['intensity'])} - {max(record['intensity'])}")
            print(f"  非零回波数: {sum(1 for x in record['intensity'] if x > 0)}")
    
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


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python parse_p62_simple.py <P62文件路径>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    parser = P62Parser(filepath)
    records = parser.parse()
    
    if records:
        stats = parser.get_statistics()
        print("\n=== 文件统计信息 ===")
        print(f"记录数: {stats['num_records']}")
        print(f"时间跨度: {stats['time_span_sec']:.2f} 秒")
        print(f"起始时间: {stats['time_start']} ({records[0]['time_str']})")
        print(f"结束时间: {stats['time_end']} ({records[-1]['time_str']})")
        print(f"方位索引范围: {stats['azimuth_min']} - {stats['azimuth_max']}")
        print(f"角度范围: {stats['angle_min']:.2f}° - {stats['angle_max']:.2f}°")
        print(f"最大量程: {stats['range_max']:.2f} 米")
        print(f"距离采样点数: {stats['num_samples']}")
        
        parser.print_sample_records(5)

