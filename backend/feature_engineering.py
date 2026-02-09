# -*- coding: utf-8 -*-
"""
烟叶风味多标签分类模型 - 特征工程模块
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

from config import (
    RANDOM_STATE, MI_THRESHOLD, RFE_N_FEATURES_TO_SELECT,
    SENSORY_COLS, CHEMICAL_COLS, PHYSICAL_COLS, APPEARANCE_COLS
)


class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols
        self.selected_features_rfe = []
        self.selected_features_mi = []
        self.key_features = []
        self.interaction_features = []
        self.final_features = []
        self.feature_report = {}
        
    def recursive_feature_elimination(self, X_train, y_train, n_features_to_select=None):
        """递归特征消除
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            n_features_to_select: 要选择的特征数量，默认使用配置文件中的 RFE_N_FEATURES_TO_SELECT
        """
        print("\n" + "=" * 60)
        print("特征工程步骤1: 递归特征消除(RFE)")
        print("=" * 60)
        
        # 对每个标签分别进行RFE
        all_selected = set()
        rfe_results = {}
        
        for label in ['Label_A', 'Label_B', 'Label_C']:
            print(f"\n处理标签: {label}")
            
            # 创建基础模型
            base_model = RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            
            # 确定选择的特征数量（优先级：参数 > 配置文件 > 自动计算）
            if n_features_to_select is not None:
                n_select = n_features_to_select
            elif RFE_N_FEATURES_TO_SELECT is not None:
                n_select = RFE_N_FEATURES_TO_SELECT
            else:
                n_select = max(5, len(self.feature_cols) // 2)
            
            print(f"  RFE筛选特征数量: {n_select} (总特征数: {len(self.feature_cols)})")
            
            # RFE
            rfe = RFE(
                estimator=base_model,
                n_features_to_select=n_select,
                step=1
            )
            
            rfe.fit(X_train, y_train[label])
            
            # 获取选中的特征
            selected = [self.feature_cols[i] for i, s in enumerate(rfe.support_) if s]
            rfe_results[label] = selected
            all_selected.update(selected)
            
            print(f"  选中特征数量: {len(selected)}")
            print(f"  选中特征: {selected}")
        
        self.selected_features_rfe = list(all_selected)
        
        print(f"\n" + "-" * 40)
        print(f"RFE筛选后特征总数: {len(self.selected_features_rfe)}")
        print(f"RFE筛选后特征列表:")
        
        # 按类别分组显示
        for category, cols in [
            ('评吸指标', SENSORY_COLS),
            ('化学成分', CHEMICAL_COLS),
            ('物理特性', PHYSICAL_COLS),
            ('外观质量', APPEARANCE_COLS)
        ]:
            selected_in_cat = [f for f in self.selected_features_rfe if f in cols]
            if selected_in_cat:
                print(f"\n  {category}: {selected_in_cat}")
        
        self.feature_report['RFE筛选结果'] = {
            '各标签选中特征': rfe_results,
            '合并后特征数量': len(self.selected_features_rfe),
            '合并后特征列表': self.selected_features_rfe
        }
        
        return self.selected_features_rfe
    
    def mutual_information_selection(self, X_train, y_train, threshold=None):
        """基于互信息的特征筛选"""
        print("\n" + "=" * 60)
        print("特征工程步骤2: 互信息特征筛选")
        print("=" * 60)
        
        if threshold is None:
            threshold = MI_THRESHOLD
        
        print(f"互信息阈值: {threshold}")
        
        # 使用RFE筛选后的特征
        X_selected = X_train[self.selected_features_rfe]
        
        # 计算每个标签的互信息
        mi_scores = {}
        all_mi_selected = set()
        
        for label in ['Label_A', 'Label_B', 'Label_C']:
            print(f"\n计算标签 {label} 的互信息值:")
            
            mi = mutual_info_classif(
                X_selected, 
                y_train[label],
                random_state=RANDOM_STATE
            )
            
            mi_df = pd.DataFrame({
                '特征': self.selected_features_rfe,
                '互信息值': mi
            }).sort_values('互信息值', ascending=False)
            
            mi_scores[label] = mi_df
            
            # 筛选超过阈值的特征
            selected = mi_df[mi_df['互信息值'] >= threshold]['特征'].tolist()
            all_mi_selected.update(selected)
            
            print(mi_df.to_string(index=False))
            print(f"\n  超过阈值的特征数量: {len(selected)}")
        
        self.selected_features_mi = list(all_mi_selected)
        
        print(f"\n" + "-" * 40)
        print(f"互信息筛选前特征数量: {len(self.selected_features_rfe)}")
        print(f"互信息筛选后特征数量: {len(self.selected_features_mi)}")
        print(f"筛选后关键品质指标: {self.selected_features_mi}")
        
        # 记录被移除的特征
        removed = set(self.selected_features_rfe) - set(self.selected_features_mi)
        if removed:
            print(f"被移除的特征: {list(removed)}")
        
        self.key_features = self.selected_features_mi.copy()
        
        self.feature_report['互信息筛选结果'] = {
            '阈值': threshold,
            '筛选前特征数量': len(self.selected_features_rfe),
            '筛选后特征数量': len(self.selected_features_mi),
            '关键品质指标': self.selected_features_mi,
            '各标签互信息值': {k: v.to_dict() for k, v in mi_scores.items()}
        }
        
        return self.selected_features_mi, mi_scores
    
    def create_interaction_features(self, X, top_n=5):
        """创建特征交互项"""
        print("\n" + "=" * 60)
        print("特征工程步骤3: 创建特征交互项")
        print("=" * 60)
        
        # 选择前top_n个关键特征进行交互
        features_for_interaction = self.key_features[:min(top_n, len(self.key_features))]
        
        print(f"用于创建交互项的特征: {features_for_interaction}")
        
        X_new = X.copy()
        self.interaction_features = []
        
        # 创建二阶交互项
        for f1, f2 in combinations(features_for_interaction, 2):
            # 乘积交互
            new_col = f"{f1}_x_{f2}"
            X_new[new_col] = X[f1] * X[f2]
            self.interaction_features.append(new_col)
            
            # 比值交互 (避免除零)
            new_col_ratio = f"{f1}_div_{f2}"
            X_new[new_col_ratio] = X[f1] / (X[f2] + 1e-8)
            self.interaction_features.append(new_col_ratio)
        
        print(f"\n创建的交互特征数量: {len(self.interaction_features)}")
        print(f"交互特征列表:")
        for i, feat in enumerate(self.interaction_features, 1):
            print(f"  {i}. {feat}")
        
        # 最终特征集
        self.final_features = self.key_features + self.interaction_features
        
        print(f"\n" + "-" * 40)
        print(f"关键品质指标子集组成:")
        print(f"  关键品质指标: {len(self.key_features)} 个")
        print(f"  特征交互项: {len(self.interaction_features)} 个")
        print(f"  总计: {len(self.final_features)} 个特征")
        
        self.feature_report['特征交互'] = {
            '用于交互的特征': features_for_interaction,
            '交互特征数量': len(self.interaction_features),
            '交互特征列表': self.interaction_features,
            '最终特征数量': len(self.final_features)
        }
        
        return X_new, self.final_features
    
    def transform_dataset(self, X):
        """转换数据集，添加交互特征"""
        X_new = X.copy()
        
        # 添加交互特征
        for feat in self.interaction_features:
            if '_x_' in feat:
                parts = feat.split('_x_')
                f1, f2 = parts[0], parts[1]
                if f1 in X.columns and f2 in X.columns:
                    X_new[feat] = X[f1] * X[f2]
            elif '_div_' in feat:
                parts = feat.split('_div_')
                f1, f2 = parts[0], parts[1]
                if f1 in X.columns and f2 in X.columns:
                    X_new[feat] = X[f1] / (X[f2] + 1e-8)
        
        return X_new[self.final_features]
    
    def run_feature_engineering(self, X_train, y_train, X_val, X_test):
        """执行完整的特征工程流程"""
        print("\n" + "=" * 80)
        print("                    特征工程流程开始")
        print("=" * 80)
        
        # RFE筛选
        self.recursive_feature_elimination(X_train, y_train)
        
        # 互信息筛选
        self.mutual_information_selection(X_train, y_train)
        
        # 创建交互特征
        X_train_new, final_features = self.create_interaction_features(X_train)
        
        # 转换验证集和测试集
        X_val_new = self.transform_dataset(X_val)
        X_test_new = self.transform_dataset(X_test)
        
        print("\n" + "=" * 80)
        print("                    特征工程完成")
        print("=" * 80)
        
        return (X_train_new[final_features], X_val_new, X_test_new, 
                final_features, self.feature_report)
