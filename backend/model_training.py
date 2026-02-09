# -*- coding: utf-8 -*-
"""
烟叶风味多标签分类模型 - 模型训练模块
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, hamming_loss
)
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import RANDOM_STATE, RF_PARAM_GRID, MODEL_DIR
import os


class MultiLabelRFModel:
    """多标签随机森林分类模型"""
    
    def __init__(self):
        self.base_model = None
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.training_report = {}
        
    def build_base_model(self):
        """构建基础模型"""
        print("\n" + "=" * 60)
        print("模型构建步骤1: 创建基础随机森林模型")
        print("=" * 60)
        
        self.base_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        print("基础模型参数:")
        for param, value in self.base_model.get_params().items():
            if param in ['n_estimators', 'max_depth', 'min_samples_split', 
                        'min_samples_leaf', 'max_features', 'random_state']:
                print(f"  {param}: {value}")
        
        self.training_report['基础模型参数'] = {
            k: v for k, v in self.base_model.get_params().items()
            if k in ['n_estimators', 'max_depth', 'min_samples_split', 
                    'min_samples_leaf', 'max_features']
        }
        
        return self.base_model
    
    def hyperparameter_tuning(self, X_train, y_train, param_grid=None):
        """超参数优化"""
        print("\n" + "=" * 60)
        print("模型构建步骤2: 超参数优化")
        print("=" * 60)
        
        if param_grid is None:
            # 使用简化的参数网格以加快搜索
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        
        print("参数搜索空间:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        # 使用Label_A作为主要优化目标
        print("\n开始网格搜索...")
        
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            param_grid=param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train['Label_A'])
        
        self.best_params = grid_search.best_params_
        
        print(f"\n最优参数:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        print(f"\n最优交叉验证得分: {grid_search.best_score_:.4f}")
        
        self.training_report['超参数优化'] = {
            '搜索空间': param_grid,
            '最优参数': self.best_params,
            '最优CV得分': grid_search.best_score_
        }
        
        return self.best_params
    
    def train_model(self, X_train, y_train):
        """训练多标签模型"""
        print("\n" + "=" * 60)
        print("模型构建步骤3: 训练多标签分类模型")
        print("=" * 60)
        
        # 使用最优参数创建模型
        optimized_rf = RandomForestClassifier(
            **self.best_params,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        # 创建多输出分类器
        self.model = MultiOutputClassifier(optimized_rf)
        
        print("开始训练多标签模型...")
        self.model.fit(X_train, y_train)
        print("模型训练完成!")
        
        # 计算特征重要性
        self.calculate_feature_importance(X_train.columns.tolist())
        
        return self.model
    
    def calculate_feature_importance(self, feature_names):
        """计算特征重要性"""
        print("\n" + "=" * 60)
        print("模型构建步骤4: 计算特征重要性")
        print("=" * 60)
        
        # 获取每个标签模型的特征重要性
        importance_dict = {}
        labels = ['Label_A', 'Label_B', 'Label_C']
        
        for i, label in enumerate(labels):
            importance = self.model.estimators_[i].feature_importances_
            importance_dict[label] = importance
        
        # 计算平均重要性
        avg_importance = np.mean(list(importance_dict.values()), axis=0)
        
        # 创建特征重要性DataFrame
        self.feature_importance = pd.DataFrame({
            '特征': feature_names,
            'Label_A重要性': importance_dict['Label_A'],
            'Label_B重要性': importance_dict['Label_B'],
            'Label_C重要性': importance_dict['Label_C'],
            '平均重要性': avg_importance
        }).sort_values('平均重要性', ascending=False)
        
        print("\n特征重要性排序 (Top 15):")
        print(self.feature_importance.head(15).to_string(index=False))
        
        self.training_report['特征重要性'] = self.feature_importance.to_dict()
        
        return self.feature_importance
    
    def validate_model(self, X_val, y_val):
        """验证模型"""
        print("\n" + "=" * 60)
        print("验证体系步骤1: 验证集评估")
        print("=" * 60)
        
        # 预测
        y_pred = self.model.predict(X_val)
        y_pred_df = pd.DataFrame(y_pred, columns=['Label_A', 'Label_B', 'Label_C'])
        
        # 计算各标签的混淆矩阵和指标
        validation_results = {}
        labels = ['Label_A', 'Label_B', 'Label_C']
        
        for i, label in enumerate(labels):
            print(f"\n{label} 验证结果:")
            
            # 混淆矩阵
            cm = confusion_matrix(y_val[label], y_pred_df[label])
            print(f"  混淆矩阵:")
            print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
            print(f"    FN={cm[1,0]}, TP={cm[1,1]}")
            
            # 计算指标
            acc = accuracy_score(y_val[label], y_pred_df[label])
            prec = precision_score(y_val[label], y_pred_df[label], zero_division=0)
            rec = recall_score(y_val[label], y_pred_df[label], zero_division=0)
            f1 = f1_score(y_val[label], y_pred_df[label], zero_division=0)
            
            print(f"  准确率: {acc:.4f}")
            print(f"  精确率: {prec:.4f}")
            print(f"  召回率: {rec:.4f}")
            print(f"  F1分数: {f1:.4f}")
            
            validation_results[label] = {
                '混淆矩阵': cm.tolist(),
                '准确率': acc,
                '精确率': prec,
                '召回率': rec,
                'F1分数': f1
            }
        
        # 计算整体Hamming Loss
        hamming = hamming_loss(y_val, y_pred_df)
        print(f"\n整体Hamming Loss: {hamming:.4f}")
        
        validation_results['Hamming_Loss'] = hamming
        self.training_report['验证集结果'] = validation_results
        
        return validation_results, y_pred_df
    
    def test_model(self, X_test, y_test):
        """测试模型"""
        print("\n" + "=" * 60)
        print("验证体系步骤2: 测试集评估")
        print("=" * 60)
        
        # 预测
        y_pred = self.model.predict(X_test)
        y_pred_df = pd.DataFrame(y_pred, columns=['Label_A', 'Label_B', 'Label_C'])
        
        # 计算各标签的指标
        test_results = {}
        labels = ['Label_A', 'Label_B', 'Label_C']
        
        print("\n测试集性能指标汇总:")
        print("-" * 60)
        print(f"{'标签':<10} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
        print("-" * 60)
        
        for i, label in enumerate(labels):
            # 混淆矩阵
            cm = confusion_matrix(y_test[label], y_pred_df[label])
            
            # 计算指标
            acc = accuracy_score(y_test[label], y_pred_df[label])
            prec = precision_score(y_test[label], y_pred_df[label], zero_division=0)
            rec = recall_score(y_test[label], y_pred_df[label], zero_division=0)
            f1 = f1_score(y_test[label], y_pred_df[label], zero_division=0)
            
            print(f"{label:<10} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")
            
            test_results[label] = {
                '混淆矩阵': cm.tolist(),
                '准确率': acc,
                '精确率': prec,
                '召回率': rec,
                'F1分数': f1
            }
        
        print("-" * 60)
        
        # 计算整体指标
        hamming = hamming_loss(y_test, y_pred_df)
        subset_acc = accuracy_score(y_test.values.tolist(), y_pred_df.values.tolist())
        
        print(f"\n整体指标:")
        print(f"  Hamming Loss: {hamming:.4f}")
        print(f"  子集准确率: {subset_acc:.4f}")
        
        test_results['Hamming_Loss'] = hamming
        test_results['子集准确率'] = subset_acc
        self.training_report['测试集结果'] = test_results
        
        return test_results, y_pred_df
    
    def predict(self, X):
        """预测新样本"""
        y_pred = self.model.predict(X)
        y_pred_df = pd.DataFrame(y_pred, columns=['Label_A', 'Label_B', 'Label_C'])
        
        # 转换为风味标签
        results = []
        for idx, row in y_pred_df.iterrows():
            labels = []
            if row['Label_A'] == 1:
                labels.append('A')
            if row['Label_B'] == 1:
                labels.append('B')
            if row['Label_C'] == 1:
                labels.append('C')
            results.append('+'.join(labels) if labels else '无标签')
        
        y_pred_df['预测风味'] = results
        return y_pred_df
    
    def save_model(self, filepath=None):
        """保存模型"""
        if filepath is None:
            filepath = os.path.join(MODEL_DIR, 'tobacco_flavor_model.pkl')
        
        model_data = {
            'model': self.model,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, filepath)
        print(f"\n模型已保存至: {filepath}")
        
        return filepath
    
    def load_model(self, filepath=None):
        """加载模型"""
        if filepath is None:
            filepath = os.path.join(MODEL_DIR, 'tobacco_flavor_model.pkl')
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.best_params = model_data['best_params']
        self.feature_importance = model_data['feature_importance']
        
        print(f"模型已从 {filepath} 加载")
        
        return self.model
