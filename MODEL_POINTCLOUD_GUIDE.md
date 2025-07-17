# 其他模型点云图生成指南

## 概述

本指南介绍如何使用新增的脚本来生成和可视化其他模型的点云预测结果。我们提供了两个主要脚本：

1. `generate_model_pointclouds.py` - 主要的点云可视化生成脚本
2. `generate_test_data.py` - 测试数据生成脚本

## 快速开始

### 1. 安装依赖

确保已安装所有必要的依赖：

```bash
pip install -r requirements.txt
pip install pillow  # 用于图像保存
```

### 2. 生成测试数据

首先生成一些测试数据来验证功能：

```bash
# 生成默认测试数据（5个扫描，每个10,000点）
python generate_test_data.py

# 自定义参数
python generate_test_data.py --output my_test_data --scans 10 --points 20000
```

这将创建以下目录结构：
```
test_data/
└── sequences/
    └── 00/
        ├── velodyne/          # 原始点云文件
        ├── labels/            # 真实标签
        ├── predictions_model_a/  # 模型A预测
        ├── predictions_model_b/  # 模型B预测
        └── predictions_model_c/  # 模型C预测
```

### 3. 生成点云可视化

#### 演示模式（最简单）

```bash
# 运行演示模式，自动生成示例数据和可视化
python generate_model_pointclouds.py --demo
```

#### 使用测试数据

```bash
# 单个模型可视化
python generate_model_pointclouds.py \
    --scan test_data/sequences/00/velodyne/000000.bin \
    --predictions test_data/sequences/00/predictions_model_a/000000.label \
    --model_names "距离模型" \
    --output results_single

# 多模型比较
python generate_model_pointclouds.py \
    --scan test_data/sequences/00/velodyne/000000.bin \
    --predictions test_data/sequences/00/predictions_model_a/000000.label \
                  test_data/sequences/00/predictions_model_b/000000.label \
                  test_data/sequences/00/predictions_model_c/000000.label \
    --model_names "距离模型" "高度模型" "随机模型" \
    --output results_comparison
```

## 详细功能说明

### 支持的文件格式

#### 点云文件
- `.bin` - KITTI格式二进制文件（推荐）
- `.npy` - NumPy数组格式

#### 预测文件
- `.label` - KITTI标签格式
- `.npy` - NumPy数组格式
- `.txt` - 文本格式

### 生成的输出

每次运行会生成以下文件：

1. **2D可视化图** (`*_2d.png`)
   - 鸟瞰图 (X-Y平面)
   - 侧视图 (X-Z平面)
   - 前视图 (Y-Z平面)
   - 距离分布直方图

2. **3D可视化图** (`*_3d.png`)
   - 交互式3D散点图
   - 语义颜色编码

3. **统计报告** (`*_stats.txt`)
   - 点云基本信息
   - 语义类别分布
   - 各类别点数和百分比

4. **模型比较图** (`*_comparison.png`)（多模型模式）
   - 并排显示不同模型的预测结果

### 高级用法

#### 自定义配置

创建自定义配置文件来修改颜色映射：

```yaml
# custom_config.yaml
color_map:
  0: [0, 0, 0]        # unlabeled - 黑色
  1: [255, 0, 0]      # car - 红色
  2: [0, 255, 0]      # bicycle - 绿色
  # ... 更多类别
```

使用自定义配置：
```bash
python generate_model_pointclouds.py \
    --scan your_scan.bin \
    --config custom_config.yaml \
    --output custom_results
```

#### 批量处理

处理整个序列：

```bash
# 创建批量处理脚本
for i in {000000..000004}; do
    python generate_model_pointclouds.py \
        --scan test_data/sequences/00/velodyne/${i}.bin \
        --predictions test_data/sequences/00/predictions_model_a/${i}.label \
        --model_names "模型A" \
        --output batch_results
done
```

## 实际使用场景

### 1. 模型开发和调试

在开发新的语义分割模型时：

```bash
# 可视化模型预测结果
python generate_model_pointclouds.py \
    --scan data/sequences/08/velodyne/000100.bin \
    --predictions my_model/predictions/000100.label \
    --model_names "我的模型" \
    --output debug_results
```

### 2. 模型性能比较

比较不同模型的性能：

```bash
# 比较多个模型
python generate_model_pointclouds.py \
    --scan data/sequences/08/velodyne/000100.bin \
    --predictions model1/pred/000100.label \
                  model2/pred/000100.label \
                  model3/pred/000100.label \
    --model_names "PointNet++" "RandLA-Net" "KPConv" \
    --output model_comparison
```

### 3. 论文和报告

生成高质量的可视化图用于学术论文：

```bash
# 生成论文用图
python generate_model_pointclouds.py \
    --scan paper_data/challenging_scene.bin \
    --predictions our_method/predictions.label \
                  baseline/predictions.label \
    --model_names "Our Method" "Baseline" \
    --output paper_figures
```

## 故障排除

### 常见问题

1. **内存不足**
   - 减少点云大小或使用更小的测试数据
   - 关闭不必要的可视化选项

2. **文件格式错误**
   - 确保点云文件是正确的KITTI格式
   - 检查预测文件的数据类型

3. **颜色显示异常**
   - 检查配置文件中的颜色映射
   - 确保预测标签在有效范围内

4. **3D可视化无法显示**
   - 确保安装了正确版本的vispy和OpenGL
   - 在Windows上可能需要额外的OpenGL驱动

### 调试模式

启用详细输出：

```bash
# 添加调试信息
python -u generate_model_pointclouds.py --demo 2>&1 | tee debug.log
```

## 扩展功能

### 添加新的可视化类型

可以修改 `generate_model_pointclouds.py` 来添加：

1. **密度图可视化**
2. **误差分析图**
3. **时序动画**
4. **交互式Web可视化**

### 集成到现有工作流

将脚本集成到模型训练流程中：

```python
# 在训练脚本中
import subprocess

def visualize_predictions(scan_path, pred_path, epoch):
    cmd = [
        'python', 'generate_model_pointclouds.py',
        '--scan', scan_path,
        '--predictions', pred_path,
        '--model_names', f'Epoch_{epoch}',
        '--output', f'training_vis/epoch_{epoch}'
    ]
    subprocess.run(cmd)
```

## 总结

这套工具提供了完整的点云可视化解决方案，支持：

- ✅ 多种文件格式
- ✅ 2D和3D可视化
- ✅ 多模型比较
- ✅ 统计分析
- ✅ 批量处理
- ✅ 自定义配置

通过这些工具，您可以轻松地生成高质量的点云可视化图，用于模型开发、性能分析和学术展示。