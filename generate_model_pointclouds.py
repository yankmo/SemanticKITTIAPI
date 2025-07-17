#!/usr/bin/env python3
"""
生成其他模型的点云图脚本
支持多种模型输出格式，生成2D和3D可视化图像
"""

import argparse
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
from auxiliary.laserscan import LaserScan, SemLaserScan
from auxiliary.laserscanvis import LaserScanVis
import vispy
from vispy import scene
from vispy.scene import visuals
from vispy.color import Colormap
import time

class ModelPointCloudGenerator:
    """用于生成和可视化模型点云预测结果的类"""
    
    def __init__(self, config_path="config/semantic-kitti.yaml"):
        """初始化生成器"""
        self.config_path = config_path
        self.load_config()
        self.setup_colors()
        
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"✓ 成功加载配置文件: {self.config_path}")
        except Exception as e:
            print(f"✗ 配置文件加载失败: {e}")
            # 使用默认配置
            self.config = self.get_default_config()
            
    def get_default_config(self):
        """获取默认配置"""
        return {
            "color_map": {
                0: [0, 0, 0],        # unlabeled - 黑色
                1: [0, 0, 255],      # car - 蓝色
                2: [245, 150, 100],  # bicycle - 橙色
                3: [245, 230, 100],  # motorcycle - 黄色
                4: [250, 80, 100],   # truck - 红色
                5: [150, 60, 30],    # other-vehicle - 棕色
                6: [255, 0, 0],      # person - 红色
                7: [30, 30, 255],    # bicyclist - 深蓝色
                8: [200, 40, 255],   # motorcyclist - 紫色
                9: [90, 30, 150],    # road - 深紫色
                10: [255, 0, 255],   # parking - 品红色
                11: [255, 150, 255], # sidewalk - 浅紫色
                12: [75, 0, 75],     # other-ground - 深紫色
                13: [75, 0, 175],    # building - 蓝紫色
                14: [0, 200, 255],   # fence - 青色
                15: [50, 120, 255],  # vegetation - 浅蓝色
                16: [0, 175, 0],     # trunk - 绿色
                17: [0, 60, 135],    # terrain - 深蓝色
                18: [80, 240, 150],  # pole - 浅绿色
                19: [150, 240, 255], # traffic-sign - 浅青色
            }
        }
        
    def setup_colors(self):
        """设置颜色映射"""
        self.color_dict = self.config.get("color_map", {})
        self.colors = np.array([self.color_dict.get(i, [128, 128, 128]) for i in range(20)]) / 255.0
        
    def load_pointcloud(self, scan_path):
        """加载点云数据"""
        try:
            if scan_path.endswith('.bin'):
                # KITTI格式的二进制文件
                points = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
                return points[:, :3], points[:, 3]  # xyz, intensity
            elif scan_path.endswith('.npy'):
                # numpy格式
                data = np.load(scan_path)
                if data.shape[1] >= 4:
                    return data[:, :3], data[:, 3]
                else:
                    return data[:, :3], np.ones(data.shape[0])
            else:
                raise ValueError(f"不支持的文件格式: {scan_path}")
        except Exception as e:
            print(f"✗ 点云加载失败: {e}")
            return None, None
            
    def load_predictions(self, pred_path):
        """加载模型预测结果"""
        try:
            if pred_path.endswith('.label'):
                # KITTI标签格式
                labels = np.fromfile(pred_path, dtype=np.uint32)
                return labels & 0xFFFF  # 取低16位作为语义标签
            elif pred_path.endswith('.npy'):
                # numpy格式
                return np.load(pred_path)
            elif pred_path.endswith('.txt'):
                # 文本格式
                return np.loadtxt(pred_path, dtype=int)
            else:
                raise ValueError(f"不支持的预测文件格式: {pred_path}")
        except Exception as e:
            print(f"✗ 预测结果加载失败: {e}")
            return None
            
    def generate_synthetic_predictions(self, points, model_type="random"):
        """生成合成预测结果用于测试"""
        n_points = points.shape[0]
        
        if model_type == "random":
            # 随机预测
            predictions = np.random.randint(0, 20, n_points)
        elif model_type == "distance_based":
            # 基于距离的预测
            distances = np.linalg.norm(points[:, :2], axis=1)  # 只考虑XY平面距离
            predictions = np.zeros(n_points, dtype=int)
            predictions[distances < 10] = 1   # 近距离 - 车辆
            predictions[(distances >= 10) & (distances < 30)] = 9  # 中距离 - 道路
            predictions[distances >= 30] = 15  # 远距离 - 植被
        elif model_type == "height_based":
            # 基于高度的预测
            heights = points[:, 2]
            predictions = np.zeros(n_points, dtype=int)
            predictions[heights < -1.5] = 9   # 低处 - 道路
            predictions[(heights >= -1.5) & (heights < 0)] = 11  # 人行道
            predictions[(heights >= 0) & (heights < 2)] = 1   # 车辆高度
            predictions[heights >= 2] = 13   # 建筑物
        elif model_type == "custom":
            # 自定义预测（用于多模型比较时的差异化）
            # 基于点的索引创建模式
            predictions = np.zeros(n_points, dtype=int)
            predictions[::5] = 1    # 每5个点标记为车辆
            predictions[1::5] = 9   # 道路
            predictions[2::5] = 15  # 植被
            predictions[3::5] = 13  # 建筑
            predictions[4::5] = 6   # 行人
        else:
            predictions = np.zeros(n_points, dtype=int)
            
        return predictions
        
    def create_2d_visualization(self, points, predictions, output_path, title="模型预测结果"):
        """创建2D鸟瞰图可视化"""
        plt.figure(figsize=(15, 10))
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. 鸟瞰图 (X-Y平面)
        scatter1 = ax1.scatter(points[:, 0], points[:, 1], c=predictions, 
                              cmap='tab20', s=0.1, alpha=0.8)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title(f'{title} - 鸟瞰图 (X-Y)')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='语义类别')
        
        # 2. 侧视图 (X-Z平面)
        scatter2 = ax2.scatter(points[:, 0], points[:, 2], c=predictions, 
                              cmap='tab20', s=0.1, alpha=0.8)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Z (m)')
        ax2.set_title(f'{title} - 侧视图 (X-Z)')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='语义类别')
        
        # 3. 前视图 (Y-Z平面)
        scatter3 = ax3.scatter(points[:, 1], points[:, 2], c=predictions, 
                              cmap='tab20', s=0.1, alpha=0.8)
        ax3.set_xlabel('Y (m)')
        ax3.set_ylabel('Z (m)')
        ax3.set_title(f'{title} - 前视图 (Y-Z)')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=ax3, label='语义类别')
        
        # 4. 距离分布图
        distances = np.linalg.norm(points[:, :2], axis=1)
        ax4.hist(distances, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_xlabel('距离 (m)')
        ax4.set_ylabel('点数量')
        ax4.set_title('点云距离分布')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 2D可视化图已保存: {output_path}")
        
    def create_3d_visualization(self, points, predictions, output_path, title="模型预测结果"):
        """创建3D可视化"""
        # 使用vispy创建3D可视化
        canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), show=True)
        view = canvas.central_widget.add_view()
        
        # 设置相机
        view.camera = 'turntable'
        view.camera.fov = 60
        view.camera.distance = 50
        
        # 创建颜色数组
        colors = np.zeros((len(points), 3))
        for i, pred in enumerate(predictions):
            if pred < len(self.colors):
                colors[i] = self.colors[pred]
            else:
                colors[i] = [0.5, 0.5, 0.5]  # 灰色作为默认颜色
                
        # 创建散点图
        scatter = visuals.Markers()
        scatter.set_data(points, edge_color=None, face_color=colors, size=2)
        view.add(scatter)
        
        # 添加坐标轴
        axis = visuals.XYZAxis(parent=view.scene)
        
        # 设置标题
        title_text = visuals.Text(title, parent=canvas.scene, color='white')
        title_text.font_size = 16
        title_text.pos = canvas.size[0] // 2, 30
        
        # 保存图像
        img = canvas.render()
        from PIL import Image
        Image.fromarray(img).save(output_path)
        canvas.close()
        print(f"✓ 3D可视化图已保存: {output_path}")
        
    def create_comparison_visualization(self, points, predictions_list, model_names, output_path):
        """创建多模型比较可视化"""
        n_models = len(predictions_list)
        fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 12))
        
        if n_models == 1:
            axes = axes.reshape(2, 1)
            
        for i, (predictions, model_name) in enumerate(zip(predictions_list, model_names)):
            # 鸟瞰图
            scatter1 = axes[0, i].scatter(points[:, 0], points[:, 1], c=predictions, 
                                        cmap='tab20', s=0.1, alpha=0.8)
            axes[0, i].set_xlabel('X (m)')
            axes[0, i].set_ylabel('Y (m)')
            axes[0, i].set_title(f'{model_name} - 鸟瞰图')
            axes[0, i].set_aspect('equal')
            axes[0, i].grid(True, alpha=0.3)
            
            # 侧视图
            scatter2 = axes[1, i].scatter(points[:, 0], points[:, 2], c=predictions, 
                                        cmap='tab20', s=0.1, alpha=0.8)
            axes[1, i].set_xlabel('X (m)')
            axes[1, i].set_ylabel('Z (m)')
            axes[1, i].set_title(f'{model_name} - 侧视图')
            axes[1, i].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 模型比较图已保存: {output_path}")
        
    def generate_statistics_report(self, points, predictions, output_path):
        """生成统计报告"""
        unique_labels, counts = np.unique(predictions, return_counts=True)
        total_points = len(points)
        
        report = f"""
点云统计报告
{'='*50}

基本信息:
- 总点数: {total_points:,}
- 点云范围:
  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}] m
  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}] m
  Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}] m

语义类别分布:
"""
        
        for label, count in zip(unique_labels, counts):
            percentage = (count / total_points) * 100
            report += f"- 类别 {label:2d}: {count:8,} 点 ({percentage:5.2f}%)\n"
            
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✓ 统计报告已保存: {output_path}")
        
    def process_single_scan(self, scan_path, pred_path=None, output_dir="output", 
                          model_name="Model", model_type="random"):
        """处理单个扫描文件"""
        print(f"\n处理扫描文件: {scan_path}")
        
        # 加载点云
        points, intensity = self.load_pointcloud(scan_path)
        if points is None:
            return False
            
        # 加载或生成预测
        if pred_path and os.path.exists(pred_path):
            predictions = self.load_predictions(pred_path)
            if predictions is None:
                print("使用合成预测数据")
                predictions = self.generate_synthetic_predictions(points, model_type)
        else:
            print("生成合成预测数据")
            predictions = self.generate_synthetic_predictions(points, model_type)
            
        # 确保预测数量匹配
        if len(predictions) != len(points):
            print(f"警告: 预测数量({len(predictions)})与点数量({len(points)})不匹配")
            predictions = predictions[:len(points)]
            
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        base_name = os.path.splitext(os.path.basename(scan_path))[0]
        
        # 生成可视化
        self.create_2d_visualization(
            points, predictions, 
            os.path.join(output_dir, f"{base_name}_{model_name}_2d.png"),
            f"{model_name} 预测结果"
        )
        
        self.create_3d_visualization(
            points, predictions,
            os.path.join(output_dir, f"{base_name}_{model_name}_3d.png"),
            f"{model_name} 预测结果"
        )
        
        # 生成统计报告
        self.generate_statistics_report(
            points, predictions,
            os.path.join(output_dir, f"{base_name}_{model_name}_stats.txt")
        )
        
        return True
        
    def process_multiple_models(self, scan_path, pred_paths, model_names, output_dir="output"):
        """处理多个模型的预测结果"""
        print(f"\n比较多个模型预测: {scan_path}")
        
        # 加载点云
        points, intensity = self.load_pointcloud(scan_path)
        if points is None:
            return False
            
        predictions_list = []
        valid_model_names = []
        
        # 加载所有模型的预测
        for pred_path, model_name in zip(pred_paths, model_names):
            if pred_path and os.path.exists(pred_path):
                predictions = self.load_predictions(pred_path)
            else:
                # 生成不同类型的合成数据
                model_types = ["random", "distance_based", "height_based"]
                model_type = model_types[len(predictions_list) % len(model_types)]
                predictions = self.generate_synthetic_predictions(points, model_type)
                
            if predictions is not None:
                if len(predictions) != len(points):
                    predictions = predictions[:len(points)]
                predictions_list.append(predictions)
                valid_model_names.append(model_name)
                
        if not predictions_list:
            print("没有有效的预测数据")
            return False
            
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成比较可视化
        base_name = os.path.splitext(os.path.basename(scan_path))[0]
        self.create_comparison_visualization(
            points, predictions_list, valid_model_names,
            os.path.join(output_dir, f"{base_name}_comparison.png")
        )
        
        # 为每个模型生成单独的可视化
        for predictions, model_name in zip(predictions_list, valid_model_names):
            self.process_single_scan(scan_path, None, output_dir, model_name, "custom")
            
        return True

def main():
    parser = argparse.ArgumentParser(description="生成其他模型的点云图")
    parser.add_argument('--scan', '-s', type=str,
                       help='点云文件路径 (.bin 或 .npy)')
    parser.add_argument('--predictions', '-p', type=str, nargs='*',
                       help='模型预测文件路径列表')
    parser.add_argument('--model_names', '-m', type=str, nargs='*',
                       help='模型名称列表')
    parser.add_argument('--output', '-o', type=str, default='model_outputs',
                       help='输出目录 (默认: model_outputs)')
    parser.add_argument('--config', '-c', type=str, default='config/semantic-kitti.yaml',
                       help='配置文件路径')
    parser.add_argument('--demo', action='store_true',
                       help='运行演示模式，生成示例数据')
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = ModelPointCloudGenerator(args.config)
    
    if args.demo:
        print("运行演示模式...")
        # 生成示例点云数据
        demo_points = np.random.randn(10000, 3) * 10
        demo_points[:, 2] += 1  # 调整高度
        
        # 保存示例数据
        os.makedirs('demo_data', exist_ok=True)
        demo_scan_path = 'demo_data/demo_scan.npy'
        np.save(demo_scan_path, demo_points)
        
        # 处理演示数据
        generator.process_multiple_models(
            demo_scan_path, 
            [None, None, None], 
            ['RandomModel', 'DistanceModel', 'HeightModel'],
            args.output
        )
    else:
        # 检查必需的参数
        if not args.scan:
            print("错误: 非演示模式需要指定 --scan 参数")
            parser.print_help()
            return
            
        # 检查输入文件
        if not os.path.exists(args.scan):
            print(f"错误: 扫描文件不存在: {args.scan}")
            return
            
        # 处理预测文件和模型名称
        predictions = args.predictions or []
        model_names = args.model_names or [f"Model_{i+1}" for i in range(len(predictions))]
        
        if len(predictions) != len(model_names):
            print("警告: 预测文件数量与模型名称数量不匹配")
            min_len = min(len(predictions), len(model_names))
            predictions = predictions[:min_len]
            model_names = model_names[:min_len]
            
        if len(predictions) > 1:
            # 多模型比较
            generator.process_multiple_models(args.scan, predictions, model_names, args.output)
        else:
            # 单模型处理
            pred_path = predictions[0] if predictions else None
            model_name = model_names[0] if model_names else "Model"
            generator.process_single_scan(args.scan, pred_path, args.output, model_name)
    
    print(f"\n✓ 处理完成! 结果保存在: {args.output}")

if __name__ == "__main__":
    main()