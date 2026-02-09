# -*- coding: utf-8 -*-
"""
烟叶风味多标签分类模型 - 报告生成模块
"""

import os
from datetime import datetime
import json

from config import OUTPUT_DIR


class ReportGenerator:
    """报告生成类"""
    
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.report_content = []
        
    def add_section(self, title, content, level=2):
        """添加章节"""
        prefix = "#" * level
        self.report_content.append(f"\n{prefix} {title}\n")
        self.report_content.append(content)
        
    def add_image(self, image_path, caption=""):
        """添加图片"""
        # 使用相对路径
        rel_path = os.path.basename(image_path)
        self.report_content.append(f"\n![{caption}]({rel_path})\n")
        if caption:
            self.report_content.append(f"*图: {caption}*\n")
    
    def add_table(self, df, caption=""):
        """添加表格"""
        if caption:
            self.report_content.append(f"\n**{caption}**\n")
        self.report_content.append(df.to_markdown(index=False))
        self.report_content.append("\n")
    
    def add_code_block(self, code, language="python"):
        """添加代码块"""
        self.report_content.append(f"\n```{language}\n{code}\n```\n")
    
    def generate_report(self, preprocessing_report, feature_report, 
                       training_report, figures, filename="analysis_report.md"):
        """生成完整报告"""
        
        # 报告标题
        self.report_content = []
        self.report_content.append("# 烟叶风味多标签分类模型分析报告\n")
        self.report_content.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.report_content.append("---\n")
        
        # 目录
        self.add_section("目录", """
1. [数据预处理](#1-数据预处理)
2. [特征工程](#2-特征工程)
3. [模型构建](#3-模型构建)
4. [验证体系](#4-验证体系)
5. [实施优化](#5-实施优化)
6. [结论与建议](#6-结论与建议)
""", level=2)
        
        # 1. 数据预处理
        self._add_preprocessing_section(preprocessing_report, figures)
        
        # 2. 特征工程
        self._add_feature_engineering_section(feature_report, figures)
        
        # 3. 模型构建
        self._add_model_building_section(training_report, figures)
        
        # 4. 验证体系
        self._add_validation_section(training_report, figures)
        
        # 5. 实施优化
        self._add_optimization_section(training_report, figures)
        
        # 6. 结论与建议
        self._add_conclusion_section(training_report)
        
        # 保存报告
        report_path = os.path.join(self.output_dir, filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report_content))
        
        print(f"\n报告已生成: {report_path}")
        
        return report_path
    
    def _add_preprocessing_section(self, report, figures):
        """添加数据预处理章节"""
        self.add_section("1. 数据预处理", "", level=2)
        
        # 1.1 数据加载
        self.add_section("1.1 数据加载", f"""
- **原始数据形状**: {report.get('原始数据形状', 'N/A')}
- **样本数量**: {report.get('样本数量', 'N/A')}
""", level=3)
        
        # 1.2 缺失值处理
        missing_info = report.get('缺失值情况', '无缺失值')
        self.add_section("1.2 缺失值处理", f"""
**缺失值情况**: {missing_info}

缺失值处理策略:
- 数值型特征: 使用均值填充
- 分类型特征: 使用众数填充
""", level=3)
        
        # 1.3 标签创建
        label_dist = report.get('标签分布', {})
        label_text = "\n".join([f"- **{k}**: {v} 个样本" for k, v in label_dist.items()])
        self.add_section("1.3 多标签创建", f"""
**标签阈值**: 50%

当某个风味标签的得分率超过50%时，认为该样品具备该风味特征。

**标签分布**:
{label_text}
""", level=3)
        
        # 添加标签分布图
        for fig in figures:
            if 'label_distribution' in fig:
                self.add_image(fig, "标签分布图")
        
        # 1.4 数据标准化
        self.add_section("1.4 数据标准化", """
采用Z-score标准化方法对所有品质指标进行标准化处理:

$$z = \\frac{x - \\mu}{\\sigma}$$

其中:
- $x$ 为原始值
- $\\mu$ 为均值
- $\\sigma$ 为标准差

标准化后，所有特征的均值接近0，标准差接近1。
""", level=3)
        
        # 1.5 数据集划分
        split_info = report.get('数据集划分', {})
        self.add_section("1.5 数据集划分", f"""
按照 **7:2:1** 的比例将数据集随机划分为训练集、验证集和测试集:

| 数据集 | 样本数量 |
|--------|----------|
| 训练集 | {split_info.get('训练集', 'N/A')} |
| 验证集 | {split_info.get('验证集', 'N/A')} |
| 测试集 | {split_info.get('测试集', 'N/A')} |
""", level=3)
        
        # 添加数据概览图
        for fig in figures:
            if 'data_overview' in fig:
                self.add_image(fig, "数据概览")
    
    def _add_feature_engineering_section(self, report, figures):
        """添加特征工程章节"""
        self.add_section("2. 特征工程", "", level=2)
        
        # 2.1 RFE特征筛选
        rfe_result = report.get('RFE筛选结果', {})
        rfe_features = rfe_result.get('合并后特征列表', [])
        
        self.add_section("2.1 递归特征消除(RFE)", f"""
使用随机森林作为基础估计器，对每个标签分别进行递归特征消除。

**RFE筛选后特征数量**: {len(rfe_features)}

**筛选后特征列表**:
{', '.join(rfe_features)}
""", level=3)
        
        # 添加RFE结果图
        for fig in figures:
            if 'rfe_results' in fig:
                self.add_image(fig, "RFE特征筛选结果")
        
        # 2.2 互信息筛选
        mi_result = report.get('互信息筛选结果', {})
        threshold = mi_result.get('阈值', 0.05)
        before_count = mi_result.get('筛选前特征数量', 'N/A')
        after_count = mi_result.get('筛选后特征数量', 'N/A')
        key_features = mi_result.get('关键品质指标', [])
        
        self.add_section("2.2 互信息特征筛选", f"""
基于互信息值对RFE筛选后的特征进行二次筛选。

**互信息阈值**: {threshold}

**筛选前特征数量**: {before_count}
**筛选后特征数量**: {after_count}

**关键品质指标**:
{', '.join(key_features)}
""", level=3)
        
        # 添加互信息图
        for fig in figures:
            if 'mi_scores' in fig:
                self.add_image(fig, "互信息值分析")
        
        # 2.3 特征交互
        interaction_result = report.get('特征交互', {})
        interaction_features = interaction_result.get('交互特征列表', [])
        final_count = interaction_result.get('最终特征数量', 'N/A')
        
        self.add_section("2.3 特征交互项生成", f"""
基于关键品质指标生成特征交互项，包括:
- **乘积交互**: $f_1 \\times f_2$
- **比值交互**: $f_1 / f_2$

**生成的交互特征数量**: {len(interaction_features)}

**关键品质指标子集组成**:
- 关键品质指标: {len(key_features)} 个
- 特征交互项: {len(interaction_features)} 个
- **总计**: {final_count} 个特征
""", level=3)
    
    def _add_model_building_section(self, report, figures):
        """添加模型构建章节"""
        self.add_section("3. 模型构建", "", level=2)
        
        # 3.1 基础模型
        base_params = report.get('基础模型参数', {})
        params_text = "\n".join([f"- **{k}**: {v}" for k, v in base_params.items()])
        
        self.add_section("3.1 随机森林基础模型", f"""
采用随机森林(Random Forest)作为基础分类器，结合MultiOutputClassifier实现多标签分类。

**基础模型参数**:
{params_text}
""", level=3)
        
        # 3.2 超参数优化
        hp_result = report.get('超参数优化', {})
        best_params = hp_result.get('最优参数', {})
        best_score = hp_result.get('最优CV得分', 'N/A')
        
        best_params_text = "\n".join([f"- **{k}**: {v}" for k, v in best_params.items()])
        
        best_score_str = f"{best_score:.4f}" if isinstance(best_score, float) else str(best_score)
        
        self.add_section("3.2 超参数优化", f"""
使用网格搜索(GridSearchCV)进行超参数优化，采用5折交叉验证。

**最优参数配置**:
{best_params_text}

**最优交叉验证得分**: {best_score_str}
""", level=3)
        
        # 3.3 特征重要性
        self.add_section("3.3 特征重要性分析", """
模型训练完成后，计算各特征对每个标签的重要性，并取平均值得到综合特征重要性排序。
""", level=3)
        
        # 添加特征重要性图
        for fig in figures:
            if 'feature_importance' in fig:
                self.add_image(fig, "特征重要性分析")
    
    def _add_validation_section(self, report, figures):
        """添加验证体系章节"""
        self.add_section("4. 验证体系", "", level=2)
        
        # 4.1 验证集评估
        val_results = report.get('验证集结果', {})
        
        self.add_section("4.1 验证集评估", """
使用验证集对模型进行评估，通过混淆矩阵和各项指标量化模型性能。
""", level=3)
        
        # 添加混淆矩阵图
        for fig in figures:
            if 'confusion_matrices' in fig:
                self.add_image(fig, "混淆矩阵")
        
        # 验证集指标表格
        if val_results:
            val_table = "| 标签 | 准确率 | 精确率 | 召回率 | F1分数 |\n"
            val_table += "|------|--------|--------|--------|--------|\n"
            for label in ['Label_A', 'Label_B', 'Label_C']:
                if label in val_results:
                    r = val_results[label]
                    val_table += f"| {label} | {r['准确率']:.4f} | {r['精确率']:.4f} | {r['召回率']:.4f} | {r['F1分数']:.4f} |\n"
            
            self.report_content.append(f"\n**验证集性能指标**:\n\n{val_table}")
        
        # 4.2 测试集评估
        test_results = report.get('测试集结果', {})
        
        self.add_section("4.2 测试集评估", """
使用测试集验证模型的泛化能力和精准预测能力。
""", level=3)
        
        # 测试集指标表格
        if test_results:
            test_table = "| 标签 | 准确率 | 精确率 | 召回率 | F1分数 |\n"
            test_table += "|------|--------|--------|--------|--------|\n"
            for label in ['Label_A', 'Label_B', 'Label_C']:
                if label in test_results:
                    r = test_results[label]
                    test_table += f"| {label} | {r['准确率']:.4f} | {r['精确率']:.4f} | {r['召回率']:.4f} | {r['F1分数']:.4f} |\n"
            
            self.report_content.append(f"\n**测试集性能指标**:\n\n{test_table}")
            
            hamming = test_results.get('Hamming_Loss', 'N/A')
            subset_acc = test_results.get('子集准确率', 'N/A')
            
            hamming_str = f"{hamming:.4f}" if isinstance(hamming, float) else str(hamming)
            subset_acc_str = f"{subset_acc:.4f}" if isinstance(subset_acc, float) else str(subset_acc)
            
            self.report_content.append(f"""
**整体指标**:
- Hamming Loss: {hamming_str}
- 子集准确率: {subset_acc_str}
""")
        
        # 添加指标对比图
        for fig in figures:
            if 'metrics_comparison' in fig:
                self.add_image(fig, "验证集与测试集性能对比")
    
    def _add_optimization_section(self, report, figures):
        """添加实施优化章节"""
        self.add_section("5. 实施优化", "", level=2)
        
        self.add_section("5.1 模型应用流程", """
1. **数据准备**: 获取待评估烟叶的品质指标数据
2. **特征处理**: 
   - 使用训练时的标准化参数对数据进行标准化
   - 提取关键品质指标
   - 生成特征交互项
3. **模型预测**: 将处理后的特征输入多标签分类模型
4. **结果输出**: 
   - 输出各风味标签的预测结果(0/1)
   - 输出综合风味预测(如A、A+B、A+B+C等)
""", level=3)
        
        self.add_section("5.2 模型优化建议", """
1. **数据层面**:
   - 增加样本数量，特别是少数类样本
   - 收集更多维度的品质指标
   
2. **特征层面**:
   - 尝试更多的特征交互方式
   - 引入领域知识构建专家特征
   
3. **模型层面**:
   - 尝试其他多标签分类算法(如Classifier Chains)
   - 使用集成学习方法提升性能
   - 针对类别不平衡问题采用过采样或欠采样策略
""", level=3)
    
    def _add_conclusion_section(self, report):
        """添加结论章节"""
        self.add_section("6. 结论与建议", "", level=2)
        
        test_results = report.get('测试集结果', {})
        
        # 计算平均指标
        avg_metrics = {'准确率': 0, '精确率': 0, '召回率': 0, 'F1分数': 0}
        count = 0
        for label in ['Label_A', 'Label_B', 'Label_C']:
            if label in test_results:
                for metric in avg_metrics:
                    avg_metrics[metric] += test_results[label].get(metric, 0)
                count += 1
        
        if count > 0:
            for metric in avg_metrics:
                avg_metrics[metric] /= count
        
        self.add_section("6.1 模型性能总结", f"""
本研究基于随机森林算法构建了烟叶风味多标签分类模型，主要结论如下:

1. **模型整体性能**:
   - 平均准确率: {avg_metrics['准确率']:.4f}
   - 平均精确率: {avg_metrics['精确率']:.4f}
   - 平均召回率: {avg_metrics['召回率']:.4f}
   - 平均F1分数: {avg_metrics['F1分数']:.4f}

2. **关键发现**:
   - 通过RFE和互信息筛选，成功识别出影响烟叶风味的关键品质指标
   - 特征交互项的引入有助于捕捉品质指标间的协同效应
   - 模型能够有效预测烟叶的多风味标签
""", level=3)
        
        self.add_section("6.2 应用建议", """
1. 在实际应用中，建议定期使用新数据对模型进行更新和验证
2. 关注特征重要性排序靠前的品质指标，这些指标对风味判断具有重要参考价值
3. 对于预测结果不确定的样品，建议结合专家评审进行综合判断
""", level=3)
