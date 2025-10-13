# PyForce
![Windows](https://img.shields.io/badge/OS-Windows-0078D6?logo=windows&logoColor=white)
![Ubuntu](https://img.shields.io/badge/OS-Ubuntu-E95420?logo=ubuntu&logoColor=white)

一个用于力传感器数据采集、可视化和保存的模块化Python包。本包设计用于使用宇立力传感器。

## 功能特性

- **简易连接**：通过TCP/IP连接力传感器的简单API
- **数据采集**：支持可配置时长的实时数据采集
- **参数配置**：设置传感器参数（采样频率、解耦矩阵、计算单位）
- **数据存储**：保存采集数据到文本文件
- **数据可视化**：生成独立和综合图表
- **模块化设计**：易于集成到其他项目中

## 安装

```bash
git clone https://github.com/Elycyx/PyForce.git
cd PyForce
pip install -e .
```

## 连接传感器
将电脑与采集卡通过网线连接。
把电脑的ipv4静态地址设为192.168.0.2，子网掩码：255.255.255.0。

## 快速导入

```python
from pyforce import ForceSensor
```

## 快速开始

### 基本用法

```python
from pyforce import ForceSensor

# 创建传感器实例
sensor = ForceSensor(ip_addr='192.168.0.108', port=4008)

# 连接传感器
if sensor.connect():
    # 查询传感器信息
    sensor.query_info()
    
    # 采集数据（按Ctrl+C停止）
    sensor.collect_data()
    
    # 保存数据
    sensor.save_data()
    
    # 生成图表
    sensor.plot_data()
    
    # 断开连接
    sensor.disconnect()
```

### 定时数据采集

```python
# 采集指定时长的数据（例如10秒）
sensor.collect_data(duration=10, print_data=True)
```

### 配置传感器设置

```python
# 设置采样频率
sensor.set_sample_rate(100)  # 100 Hz

# 设置计算单位
sensor.set_compute_unit("MVPV")

# 设置解耦矩阵（使用你的校准值）
decouple_matrix = "(0.272516,-62.753809,...)...\r\n"
sensor.set_decouple_matrix(decouple_matrix)
```

### 编程访问数据

```python
# 获取最新数据点
latest = sensor.get_latest_data()
print(f"Fx: {latest['fx']} N, Fz: {latest['fz']} N")

# 获取所有采集的数据
all_data = sensor.get_all_data()
print(f"总数据点数: {len(all_data['timestamps'])}")
```

### 自定义数据处理

```python
import numpy as np

# 采集数据
sensor.collect_data(duration=5, print_data=False)

# 获取所有数据
data = sensor.get_all_data()

# 计算统计信息
print(f"Fz平均值: {np.mean(data['fz']):.3f} N")
print(f"Fz最大值: {np.max(data['fz']):.3f} N")
```

## API参考

### ForceSensor类

#### 构造函数

```python
ForceSensor(ip_addr='192.168.0.108', port=4008)
```

- `ip_addr`: 传感器IP地址
- `port`: 传感器端口

#### 方法

**连接管理**
- `connect()` → bool: 连接到传感器
- `disconnect()`: 断开传感器连接

**传感器配置**
- `query_info()` → Dict: 查询传感器信息
- `set_sample_rate(rate)` → bool: 设置采样频率（Hz）
- `set_compute_unit(unit)` → bool: 设置计算单位
- `set_decouple_matrix(matrix)` → bool: 设置解耦矩阵
- `set_data_format(format_str)` → bool: 设置数据上传格式

**数据采集**
- `collect_data(duration=None, print_data=True)` → bool: 采集传感器数据
  - `duration`: 采集时长（秒），None表示手动停止
  - `print_data`: 是否打印实时数据
- `start_stream()` → bool: 开始数据流
- `stop_stream()` → bool: 停止数据流
- `clear_data()`: 清空数据缓冲区

**数据访问**
- `get_latest_data()` → Dict: 获取最新数据点
- `get_all_data()` → Dict: 获取所有采集的数据
- `parse_data(data)` → Tuple: 解析原始传感器数据包

**数据存储与可视化**
- `save_data(filename=None)` → str: 保存数据到文件
- `plot_data(save_charts=True, show_charts=True, chart_prefix=None)` → List: 生成图表

## 数据格式

### 采集的数据结构

每个数据点包含：
- `time`: 时间戳（秒）
- `fx`: X方向力（N）
- `fy`: Y方向力（N）
- `fz`: Z方向力（N）
- `mx`: 绕X轴力矩（Nm）
- `my`: 绕Y轴力矩（Nm）
- `mz`: 绕Z轴力矩（Nm）

### 文件输出

数据以制表符分隔格式保存：

```
Time(s)    Fx(N)    Fy(N)    Fz(N)    Mx(Nm)    My(Nm)    Mz(Nm)
0.0000     1.234    2.345    3.456    0.123     0.234     0.345
...
```

## 示例

查看 `examples/example_usage.py` 获取详细示例：

1. **基本用法**：简单的数据采集和可视化
2. **定时采集**：采集指定时长的数据
3. **自定义设置**：配置传感器参数
4. **实时处理**：不存储所有数据的处理
5. **多次会话**：采集和比较多个数据集
6. **数据访问**：访问和分析采集的数据
7. **回调集成**：使用回调处理数据

运行示例：

```bash
python examples/example_usage.py
```

## 故障排除

### 连接问题

- 检查IP地址和端口是否正确
- 确保传感器已开机并连接到网络
- 使用 `ping` 验证网络连接

### 数据采集问题

- 如果没有接收到数据，检查传感器配置
- 验证数据格式设置是否与传感器输出匹配
- 检查防火墙设置

### 可视化问题

- 如果图表不显示，可能没有GUI（使用 `show_charts=False`）
- 即使显示失败，图表也会保存到 `charts/` 目录
- 确保matplotlib后端与你的系统兼容

## 文件结构

```
PyForce/
├── pyforce/              # 主包
│   ├── __init__.py      # 包初始化
│   └── sensor.py        # ForceSensor类
├── examples/            # 使用示例
│   └── example_usage.py
├── setup.py            # 安装配置
├── pyproject.toml      # 项目元数据
├── requirements.txt    # Python依赖
├── README.md          # 英文文档
├── README_CN.md       # 中文文档（本文件）
├── .gitignore         # Git忽略规则
├── data/              # 数据文件（自动创建，被忽略）
└── charts/            # 图表图片（自动创建，被忽略）
```

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。


