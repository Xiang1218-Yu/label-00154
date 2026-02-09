# -*- coding: utf-8 -*-
"""
烟叶风味多标签分类模型 - 主程序
基于随机森林算法的多标签分类模型

功能:
1. 数据预处理 - 缺失值处理、数据标准化
2. 特征工程 - RFE筛选、互信息筛选、特征交互
3. 模型构建 - 随机森林多标签分类
4. 验证体系 - 混淆矩阵、性能指标评估
5. 实施优化 - 模型保存、预测应用
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')

from config import DATA_FILE, OUTPUT_DIR, MODEL_DIR
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import MultiLabelRFModel
from visualization import Visualizer
from report_generator import ReportGenerator


def check_data_file():
    """检查数据文件是否存在"""
    if not os.path.exists(DATA_FILE):
        print(f"错误: 数据文件不存在: {DATA_FILE}")
        print(f"请将数据文件 'peifutf8.csv' 放置到 {os.path.dirname(DATA_FILE)} 目录下")
        return False
    return True


def main():
    """主函数"""
    print("=" * 80)
    print("        烟叶风味多标签分类模型 - 基于随机森林算法")
    print("=" * 80)
    print()
    
    # 检查数据文件
    if not check_data_file():
        # 如果没有数据文件，生成示例数据用于演示
        print("\n正在生成示例数据用于演示...")
        generate_sample_data()
    
    # 初始化各模块
    preprocessor = DataPreprocessor()
    visualizer = Visualizer()
    report_generator = ReportGenerator()
    
    # ========== 步骤1: 数据预处理 ==========
    print("\n" + "=" * 80)
    print("                    第一阶段: 数据预处理")
    print("=" * 80)
    
    datasets, feature_cols, scaler, preprocessing_report = preprocessor.run_preprocessing()
    X_train, X_val, X_test, y_train, y_val, y_test = datasets
    
    # 生成数据预处理相关图表
    visualizer.plot_label_distribution(y_train, "训练集标签分布", "label_distribution.png")
    visualizer.plot_data_overview(preprocessor.processed_data, feature_cols, "data_overview.png")
    
    # ========== 步骤2: 特征工程 ==========
    print("\n" + "=" * 80)
    print("                    第二阶段: 特征工程")
    print("=" * 80)
    
    feature_engineer = FeatureEngineer(feature_cols)
    (X_train_fe, X_val_fe, X_test_fe, 
     final_features, feature_report) = feature_engineer.run_feature_engineering(
        X_train, y_train, X_val, X_test
    )
    
    # 生成特征工程相关图表
    visualizer.plot_rfe_results(
        feature_engineer.selected_features_rfe, 
        feature_cols, 
        "rfe_results.png"
    )
    
    # 如果有互信息分数，绘制互信息图
    if hasattr(feature_engineer, 'mi_scores_cache'):
        visualizer.plot_mi_scores(feature_engineer.mi_scores_cache, "mi_scores.png")
    
    visualizer.plot_feature_correlation(X_train_fe, final_features, "feature_correlation.png")
    
    # ========== 步骤3: 模型构建 ==========
    print("\n" + "=" * 80)
    print("                    第三阶段: 模型构建")
    print("=" * 80)
    
    model = MultiLabelRFModel()
    model.build_base_model()
    model.hyperparameter_tuning(X_train_fe, y_train)
    model.train_model(X_train_fe, y_train)
    
    # 输出当前使用的模型
    print(f"\n当前使用的模型: RandomForestClassifier (MultiOutputClassifier)")
    print(f"模型最优参数: {model.best_params}")
    
    # 生成特征重要性图
    visualizer.plot_feature_importance(model.feature_importance, top_n=15, 
                                       filename="feature_importance.png")
    
    # ========== 步骤4: 验证体系 ==========
    print("\n" + "=" * 80)
    print("                    第四阶段: 验证体系")
    print("=" * 80)
    
    # 验证集评估
    val_results, y_val_pred = model.validate_model(X_val_fe, y_val)
    
    # 测试集评估
    test_results, y_test_pred = model.test_model(X_test_fe, y_test)
    
    # 生成验证相关图表
    visualizer.plot_confusion_matrices(y_test, y_test_pred, "confusion_matrices.png")
    visualizer.plot_metrics_comparison(val_results, test_results, "metrics_comparison.png")
    
    # ========== 步骤5: 实施优化 ==========
    print("\n" + "=" * 80)
    print("                    第五阶段: 实施优化")
    print("=" * 80)
    
    # 保存模型
    model_path = model.save_model()
    
    # 保存特征工程参数
    import joblib
    fe_params = {
        'scaler': scaler,
        'feature_cols': feature_cols,
        'final_features': final_features,
        'key_features': feature_engineer.key_features,
        'interaction_features': feature_engineer.interaction_features
    }
    fe_path = os.path.join(MODEL_DIR, 'feature_engineering_params.pkl')
    joblib.dump(fe_params, fe_path)
    print(f"特征工程参数已保存至: {fe_path}")
    
    # ========== 生成报告 ==========
    print("\n" + "=" * 80)
    print("                    生成分析报告")
    print("=" * 80)
    
    # 合并训练报告
    training_report = model.training_report
    
    # 生成报告
    figures = visualizer.generate_all_figures()
    report_path = report_generator.generate_report(
        preprocessing_report,
        feature_report,
        training_report,
        figures
    )
    
    # ========== 完成 ==========
    print("\n" + "=" * 80)
    print("                    分析完成")
    print("=" * 80)
    print(f"\n输出目录: {OUTPUT_DIR}")
    print(f"模型文件: {model_path}")
    print(f"分析报告: {report_path}")
    print("\n生成的图表:")
    for fig in figures:
        print(f"  - {os.path.basename(fig)}")
    
    return model, feature_engineer, scaler


def generate_sample_data():
    """生成示例数据"""
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    n_samples = 150
    
    # 生成示例数据
    data = {
        '样品编号': [f'S{i:03d}' for i in range(1, n_samples + 1)],
        '省份': np.random.choice(['云南', '贵州', '四川', '湖南', '福建'], n_samples),
        '品种': np.random.choice(['K326', 'NC89', 'NC82', 'G80'], n_samples),
        '等级': np.random.choice(['C1F', 'C2F', 'C3F', 'B1F', 'B2F'], n_samples),
        '部位': np.random.choice(['上部', '中部', '下部'], n_samples),
        '香型': np.random.choice(['清香型', '浓香型', '中间香型'], n_samples),
        '样品类别': np.random.choice(['原料', '成品'], n_samples),
        'A得分率': np.random.uniform(0.2, 1.0, n_samples),
        'B得分率': np.random.uniform(0.1, 0.8, n_samples),
        'C得分率': np.random.uniform(0.1, 0.6, n_samples),
    }
    
    # 评吸指标
    sensory_cols = ['香型风格', '彰显程度', '香气质', '香气量', '透发性', '杂气', 
                    '刺激性', '余味', '甜感', '浓度', '细腻程度', '劲头', 
                    '可用性赋分', '烟气特征', '香气特征', '口感特征']
    for col in sensory_cols:
        data[col] = np.random.uniform(50, 100, n_samples)
    
    # 化学成分
    data['总植物碱'] = np.random.uniform(1.5, 4.0, n_samples)
    data['总氮'] = np.random.uniform(1.5, 3.5, n_samples)
    data['还原糖'] = np.random.uniform(15, 30, n_samples)
    data['总糖'] = np.random.uniform(20, 40, n_samples)
    data['钾'] = np.random.uniform(1.5, 3.5, n_samples)
    data['氯'] = np.random.uniform(0.2, 1.0, n_samples)
    data['淀粉'] = np.random.uniform(2, 8, n_samples)
    
    # 物理特性
    data['厚度'] = np.random.uniform(0.08, 0.15, n_samples)
    data['密度'] = np.random.uniform(0.3, 0.5, n_samples)
    data['单叶重'] = np.random.uniform(8, 15, n_samples)
    data['平衡含水率'] = np.random.uniform(10, 14, n_samples)
    data['拉力'] = np.random.uniform(0.5, 1.5, n_samples)
    data['伸长量'] = np.random.uniform(1, 3, n_samples)
    data['填充值'] = np.random.uniform(3, 5, n_samples)
    data['含梗率'] = np.random.uniform(20, 35, n_samples)
    
    # 外观质量
    data['叶片结构'] = np.random.uniform(60, 90, n_samples)
    data['成熟度'] = np.random.uniform(70, 95, n_samples)
    data['油分'] = np.random.uniform(60, 90, n_samples)
    data['色度'] = np.random.uniform(65, 95, n_samples)
    data['身份'] = np.random.uniform(60, 85, n_samples)
    data['颜色'] = np.random.uniform(70, 95, n_samples)
    
    df = pd.DataFrame(data)
    
    # 保存数据
    from config import DATA_DIR
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(DATA_FILE, index=False, encoding='utf-8')
    print(f"示例数据已生成: {DATA_FILE}")


if __name__ == "__main__":
    model, feature_engineer, scaler = main()
