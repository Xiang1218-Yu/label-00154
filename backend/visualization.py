# -*- coding: utf-8 -*-
"""
烟叶风味多标签分类模型 - 可视化模块
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

from config import OUTPUT_DIR, FIGURE_DPI, FIGURE_SIZE

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    """可视化类"""
    
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.figures = []
        
    def plot_label_distribution(self, y_data, title="标签分布", filename="label_distribution.png"):
        """绘制标签分布图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        labels = ['Label_A', 'Label_B', 'Label_C']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for i, (label, color) in enumerate(zip(labels, colors)):
            counts = y_data[label].value_counts().sort_index()
            axes[i].bar(['负类(0)', '正类(1)'], counts.values, color=color, alpha=0.7)
            axes[i].set_title(f'{label} 分布')
            axes[i].set_ylabel('样本数量')
            
            # 添加数值标签
            for j, v in enumerate(counts.values):
                axes[i].text(j, v + 0.5, str(v), ha='center', fontsize=12)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        self.figures.append(filepath)
        print(f"图表已保存: {filepath}")
        
        return filepath
    
    def plot_feature_importance(self, feature_importance, top_n=15, 
                                filename="feature_importance.png"):
        """绘制特征重要性图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # 取前top_n个特征
        top_features = feature_importance.head(top_n)
        
        # 平均重要性
        ax = axes[0, 0]
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]
        bars = ax.barh(range(top_n), top_features['平均重要性'].values[::-1], color=colors)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features['特征'].values[::-1])
        ax.set_xlabel('重要性')
        ax.set_title('平均特征重要性 (Top {})'.format(top_n))
        
        # 各标签重要性
        for idx, (label, ax) in enumerate(zip(['Label_A', 'Label_B', 'Label_C'], 
                                              [axes[0, 1], axes[1, 0], axes[1, 1]])):
            col_name = f'{label}重要性'
            sorted_df = feature_importance.sort_values(col_name, ascending=False).head(top_n)
            
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))[::-1]
            ax.barh(range(top_n), sorted_df[col_name].values[::-1], color=colors)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(sorted_df['特征'].values[::-1])
            ax.set_xlabel('重要性')
            ax.set_title(f'{label} 特征重要性 (Top {top_n})')
        
        plt.suptitle('特征重要性分析', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        self.figures.append(filepath)
        print(f"图表已保存: {filepath}")
        
        return filepath
    
    def plot_confusion_matrices(self, y_true, y_pred, filename="confusion_matrices.png"):
        """绘制混淆矩阵"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        labels = ['Label_A', 'Label_B', 'Label_C']
        titles = ['风味A', '风味B', '风味C']
        
        for i, (label, title) in enumerate(zip(labels, titles)):
            cm = confusion_matrix(y_true[label], y_pred[label])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['预测负类', '预测正类'],
                       yticklabels=['实际负类', '实际正类'])
            axes[i].set_title(f'{title} 混淆矩阵')
            axes[i].set_ylabel('实际值')
            axes[i].set_xlabel('预测值')
        
        plt.suptitle('各标签混淆矩阵', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        self.figures.append(filepath)
        print(f"图表已保存: {filepath}")
        
        return filepath
    
    def plot_metrics_comparison(self, val_results, test_results, 
                                filename="metrics_comparison.png"):
        """绘制验证集和测试集指标对比"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        labels = ['Label_A', 'Label_B', 'Label_C']
        metrics = ['准确率', '精确率', '召回率', 'F1分数']
        
        x = np.arange(len(labels))
        width = 0.35
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            val_values = [val_results[label][metric] for label in labels]
            test_values = [test_results[label][metric] for label in labels]
            
            bars1 = ax.bar(x - width/2, val_values, width, label='验证集', color='#3498db', alpha=0.8)
            bars2 = ax.bar(x + width/2, test_values, width, label='测试集', color='#e74c3c', alpha=0.8)
            
            ax.set_ylabel(metric)
            ax.set_title(f'{metric}对比')
            ax.set_xticks(x)
            ax.set_xticklabels(['风味A', '风味B', '风味C'])
            ax.legend()
            ax.set_ylim(0, 1.1)
            
            # 添加数值标签
            for bar in bars1:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
            
            for bar in bars2:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('验证集与测试集性能指标对比', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        self.figures.append(filepath)
        print(f"图表已保存: {filepath}")
        
        return filepath
    
    def plot_feature_correlation(self, X, feature_cols, filename="feature_correlation.png"):
        """绘制特征相关性热力图"""
        # 选择前20个特征
        cols_to_plot = feature_cols[:min(20, len(feature_cols))]
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        corr_matrix = X[cols_to_plot].corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='RdBu_r', center=0, ax=ax,
                   square=True, linewidths=0.5,
                   annot_kws={'size': 8})
        
        ax.set_title('特征相关性热力图', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        self.figures.append(filepath)
        print(f"图表已保存: {filepath}")
        
        return filepath
    
    def plot_mi_scores(self, mi_scores, filename="mi_scores.png"):
        """绘制互信息值图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        labels = ['Label_A', 'Label_B', 'Label_C']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for i, (label, color) in enumerate(zip(labels, colors)):
            mi_df = mi_scores[label].head(15)
            
            axes[i].barh(range(len(mi_df)), mi_df['互信息值'].values[::-1], color=color, alpha=0.7)
            axes[i].set_yticks(range(len(mi_df)))
            axes[i].set_yticklabels(mi_df['特征'].values[::-1])
            axes[i].set_xlabel('互信息值')
            axes[i].set_title(f'{label} 互信息值 (Top 15)')
            axes[i].axvline(x=0.05, color='red', linestyle='--', label='阈值=0.05')
            axes[i].legend()
        
        plt.suptitle('各标签特征互信息值', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        self.figures.append(filepath)
        print(f"图表已保存: {filepath}")
        
        return filepath
    
    def plot_data_overview(self, data, feature_cols, filename="data_overview.png"):
        """绘制数据概览图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # 1. 特征分布箱线图
        ax = axes[0, 0]
        cols_to_plot = feature_cols[:min(10, len(feature_cols))]
        data[cols_to_plot].boxplot(ax=ax, rot=45)
        ax.set_title('主要特征分布箱线图')
        ax.set_ylabel('标准化值')
        
        # 2. 样本类别分布
        ax = axes[0, 1]
        if '样品类别' in data.columns:
            data['样品类别'].value_counts().plot(kind='bar', ax=ax, color='#3498db', alpha=0.7)
            ax.set_title('样品类别分布')
            ax.set_ylabel('数量')
            ax.tick_params(axis='x', rotation=45)
        
        # 3. 省份分布
        ax = axes[1, 0]
        if '省份' in data.columns:
            data['省份'].value_counts().head(10).plot(kind='bar', ax=ax, color='#2ecc71', alpha=0.7)
            ax.set_title('省份分布 (Top 10)')
            ax.set_ylabel('数量')
            ax.tick_params(axis='x', rotation=45)
        
        # 4. 香型分布
        ax = axes[1, 1]
        if '香型' in data.columns:
            data['香型'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')
            ax.set_title('香型分布')
            ax.set_ylabel('')
        
        plt.suptitle('数据概览', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        self.figures.append(filepath)
        print(f"图表已保存: {filepath}")
        
        return filepath
    
    def plot_rfe_results(self, rfe_features, all_features, filename="rfe_results.png"):
        """绘制RFE筛选结果"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 创建特征选择状态
        feature_status = ['选中' if f in rfe_features else '未选中' for f in all_features]
        
        colors = ['#2ecc71' if s == '选中' else '#e74c3c' for s in feature_status]
        
        y_pos = range(len(all_features))
        ax.barh(y_pos, [1] * len(all_features), color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_features)
        ax.set_xlabel('特征选择状态')
        ax.set_title(f'RFE特征筛选结果 (选中 {len(rfe_features)}/{len(all_features)} 个特征)')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#2ecc71', label='选中'),
                         Patch(facecolor='#e74c3c', label='未选中')]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        self.figures.append(filepath)
        print(f"图表已保存: {filepath}")
        
        return filepath
    
    def generate_all_figures(self):
        """返回所有生成的图表路径"""
        return self.figures
