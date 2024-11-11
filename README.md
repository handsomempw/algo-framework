# 算法框架系统 (Algorithm Framework System)

一个基于 Python 开发的高性能视频算法处理框架，支持实时视频流分析和多任务并行处理。该框架采用模块化设计，可以灵活扩展不同类型的视觉算法。

## ✨ 主要特性

- 🎯 模块化设计，支持灵活扩展不同算法
- 🎥 实时视频流处理能力
- 🚀 多任务并行处理
- 📊 内置性能监控
- 🔄 支持动态任务配置更新
- 📝 完善的日志记录
- 🛡️ 异常处理和自动恢复机制
- 🔌 标准化的算法接口

## 🎯 系统架构

系统主要包含以下核心组件:

- **AlgoSystem**: 系统主控制器，负责协调各个组件的工作
- **TaskManager**: 任务管理器，处理任务的生命周期
- **AlgorithmManager**: 算法管理器，负责算法的加载和实例化
- **BaseAlgorithm**: 算法基类，提供标准化的算法接口

## 💡 已实现的算法

### 1. 黑名单人脸检测
- 基于人脸识别API的实时检测
- 支持黑名单库动态更新
- 可配置的匹配阈值和检测参数

### 2. 垃圾检测
- 基于Triton推理服务的目标检测
- ROI区域过滤
- 支持多类别垃圾识别
- 结果图片和事件数据保存

## 🔧 项目结构

```
algo_framework/
├── core/                   # 核心组件
│   ├── algorithm_manager.py
│   ├── task_manager.py
│   ├── resource_manager.py
│   └── output_manager.py
├── algorithms/             # 算法实现
│   ├── base_algorithm.py
│   ├── blacklist/
│       ├── blacklist_algorithm.py
│       └── face_api.py
│   └── others/
├── __init__.py
└── main.py                # 入口文件
```

## 🚀 快速开始

```bash
python main.py
```

## 📝 最近更新

- ✨ 新增垃圾检测算法
- 🔧 完善算法基类注释
- 📝 优化测试的启停逻辑
- 🎨 添加结果保存功能
- 🐛 修复性能监控问题
