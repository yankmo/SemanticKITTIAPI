#!/usr/bin/env python3
"""
生成测试数据脚本
为演示和测试目的创建示例点云数据
"""

import numpy as np
import os
import argparse

def generate_test_pointcloud(n_points=10000, scene_type="urban"):
    """生成测试点云数据"""
    
    if scene_type == "urban":
        # 城市场景：道路、建筑、车辆
        points_per_class = n_points // 3
        
        # 道路 (平面，z接近0)
        road_points = np.random.uniform(-50, 50, (points_per_class, 2))
        road_z = np.random.normal(-1.5, 0.2, points_per_class)
        road_xyz = np.column_stack([road_points, road_z])
        road_labels = np.full(points_per_class, 9)  # road class
        
        # 建筑物 (高结构)
        building_x = np.random.uniform(-30, 30, points_per_class)
        building_y = np.random.uniform(-30, 30, points_per_class)
        building_z = np.random.uniform(0, 15, points_per_class)
        building_xyz = np.column_stack([building_x, building_y, building_z])
        building_labels = np.full(points_per_class, 13)  # building class
        
        # 车辆 (中等高度) - 补齐剩余点数
        remaining_points = n_points - 2 * points_per_class
        vehicle_x = np.random.uniform(-20, 20, remaining_points)
        vehicle_y = np.random.uniform(-20, 20, remaining_points)
        vehicle_z = np.random.uniform(-1, 2, remaining_points)
        vehicle_xyz = np.column_stack([vehicle_x, vehicle_y, vehicle_z])
        vehicle_labels = np.full(remaining_points, 1)  # car class
        
        points = np.vstack([road_xyz, building_xyz, vehicle_xyz])
        labels = np.hstack([road_labels, building_labels, vehicle_labels])
        
    elif scene_type == "highway":
        # 高速公路场景
        points_per_class = n_points // 2
        
        # 道路中心线
        road_center = np.linspace(-100, 100, points_per_class)
        road_y = np.random.normal(0, 5, points_per_class)
        road_z = np.random.normal(-1.5, 0.1, points_per_class)
        road_xyz = np.column_stack([road_center, road_y, road_z])
        road_labels = np.full(points_per_class, 9)
        
        # 植被 (道路两侧) - 补齐剩余点数
        remaining_points = n_points - points_per_class
        veg_x = np.random.uniform(-100, 100, remaining_points)
        veg_y = np.random.choice([-20, 20], remaining_points) + np.random.normal(0, 5, remaining_points)
        veg_z = np.random.uniform(0, 10, remaining_points)
        veg_xyz = np.column_stack([veg_x, veg_y, veg_z])
        veg_labels = np.full(remaining_points, 15)  # vegetation
        
        points = np.vstack([road_xyz, veg_xyz])
        labels = np.hstack([road_labels, veg_labels])
        
    else:  # random
        points = np.random.randn(n_points, 3) * 20
        labels = np.random.randint(0, 20, n_points)
    
    # 添加强度信息 - 确保与点数匹配
    intensity = np.random.uniform(0, 1, len(points))
    
    return points, intensity, labels

def save_kitti_format(points, intensity, labels, scan_path, label_path):
    """保存为KITTI格式"""
    # 保存点云 (.bin格式)
    scan_data = np.column_stack([points, intensity]).astype(np.float32)
    scan_data.tofile(scan_path)
    
    # 保存标签 (.label格式)
    labels.astype(np.uint32).tofile(label_path)

def create_test_dataset(output_dir="test_data", n_scans=5, n_points=10000):
    """创建完整的测试数据集"""
    
    # 创建目录结构
    sequences_dir = os.path.join(output_dir, "sequences", "00")
    velodyne_dir = os.path.join(sequences_dir, "velodyne")
    labels_dir = os.path.join(sequences_dir, "labels")
    
    os.makedirs(velodyne_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    scene_types = ["urban", "highway", "random"]
    
    print(f"生成 {n_scans} 个测试扫描文件...")
    
    for i in range(n_scans):
        # 选择场景类型
        scene_type = scene_types[i % len(scene_types)]
        
        # 生成数据
        points, intensity, labels = generate_test_pointcloud(n_points, scene_type)
        
        # 文件名
        scan_file = f"{i:06d}.bin"
        label_file = f"{i:06d}.label"
        
        scan_path = os.path.join(velodyne_dir, scan_file)
        label_path = os.path.join(labels_dir, label_file)
        
        # 保存文件
        save_kitti_format(points, intensity, labels, scan_path, label_path)
        
        print(f"  ✓ 生成扫描 {i:06d} ({scene_type} 场景)")
    
    # 创建不同模型的预测目录
    models = ["model_a", "model_b", "model_c"]
    
    for model_name in models:
        model_pred_dir = os.path.join(sequences_dir, f"predictions_{model_name}")
        os.makedirs(model_pred_dir, exist_ok=True)
        
        print(f"为 {model_name} 生成预测结果...")
        
        for i in range(n_scans):
            # 加载原始点云以获取点数
            scan_path = os.path.join(velodyne_dir, f"{i:06d}.bin")
            scan_data = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
            n_points_actual = scan_data.shape[0]
            
            # 生成不同的预测策略
            if model_name == "model_a":
                # 基于距离的预测
                points = scan_data[:, :3]
                distances = np.linalg.norm(points, axis=1)
                predictions = np.zeros(n_points_actual, dtype=np.uint32)
                predictions[distances < 10] = 1   # 近距离 - 车辆
                predictions[(distances >= 10) & (distances < 30)] = 9  # 中距离 - 道路
                predictions[distances >= 30] = 15  # 远距离 - 植被
                
            elif model_name == "model_b":
                # 基于高度的预测
                heights = scan_data[:, 2]
                predictions = np.zeros(n_points_actual, dtype=np.uint32)
                predictions[heights < -1] = 9     # 低处 - 道路
                predictions[(heights >= -1) & (heights < 1)] = 1   # 车辆高度
                predictions[(heights >= 1) & (heights < 5)] = 6    # 人员高度
                predictions[heights >= 5] = 13   # 建筑物
                
            else:  # model_c
                # 随机预测（模拟较差的模型）
                predictions = np.random.randint(0, 20, n_points_actual, dtype=np.uint32)
            
            # 保存预测结果
            pred_path = os.path.join(model_pred_dir, f"{i:06d}.label")
            predictions.tofile(pred_path)
        
        print(f"  ✓ {model_name} 预测结果已生成")
    
    print(f"\n✓ 测试数据集创建完成: {output_dir}")
    print(f"  - {n_scans} 个扫描文件")
    print(f"  - {len(models)} 个模型的预测结果")
    print(f"  - 每个扫描约 {n_points:,} 个点")

def main():
    parser = argparse.ArgumentParser(description="生成测试点云数据")
    parser.add_argument('--output', '-o', type=str, default='test_data',
                       help='输出目录 (默认: test_data)')
    parser.add_argument('--scans', '-s', type=int, default=5,
                       help='生成的扫描文件数量 (默认: 5)')
    parser.add_argument('--points', '-p', type=int, default=10000,
                       help='每个扫描的点数 (默认: 10000)')
    
    args = parser.parse_args()
    
    create_test_dataset(args.output, args.scans, args.points)

if __name__ == "__main__":
    main()