# -*- coding: utf-8 -*-
"""
烟叶风味多标签分类模型 - 数据预处理模块
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_FILE, LABEL_THRESHOLD, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    RANDOM_STATE, CATEGORICAL_COLS, LABEL_COLS, ALL_QUALITY_COLS
)


class DataPreprocessor:
    """数据预处理类"""
    
    def __init__(self, data_path=None):
        self.data_path = data_path or DATA_FILE
        self.raw_data = None
        self.processed_data = None
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.preprocessing_report = {}
        
    def load_data(self):
        """加载数据"""
        print("=" * 60)
        print("步骤1: 数据加载")
        print("=" * 60)
        
        try:
            self.raw_data = pd.read_csv(self.data_path, encoding='utf-8')
        except:
            self.raw_data = pd.read_csv(self.data_path, encoding='gbk')
        
        print(f"数据文件: {self.data_path}")
        print(f"原始数据形状: {self.raw_data.shape}")
        print(f"样本数量: {self.raw_data.shape[0]}")
        print(f"特征数量: {self.raw_data.shape[1]}")
        
        self.preprocessing_report['原始数据形状'] = self.raw_data.shape
        self.preprocessing_report['样本数量'] = self.raw_data.shape[0]
        
        return self.raw_data
    
    def analyze_missing_values(self):
        """分析缺失值"""
        print("\n" + "=" * 60)
        print("步骤2: 缺失值分析")
        print("=" * 60)
        
        missing_info = self.raw_data.isnull().sum()
        missing_cols = missing_info[missing_info > 0]
        
        if len(missing_cols) > 0:
            print("\n存在缺失值的列:")
            missing_df = pd.DataFrame({
                '缺失数量': missing_cols,
                '缺失比例(%)': (missing_cols / len(self.raw_data) * 100).round(2)
            })
            print(missing_df)
            self.preprocessing_report['缺失值情况'] = missing_df.to_dict()
        else:
            print("数据中无缺失值")
            self.preprocessing_report['缺失值情况'] = "无缺失值"
        
        return missing_info
    
    def handle_missing_values(self):
        """处理缺失值"""
        print("\n" + "=" * 60)
        print("步骤3: 缺失值处理")
        print("=" * 60)
        
        self.processed_data = self.raw_data.copy()
        
        # 识别数值型列
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 对数值型列使用均值填充
        for col in numeric_cols:
            if self.processed_data[col].isnull().sum() > 0:
                mean_val = self.processed_data[col].mean()
                self.processed_data[col].fillna(mean_val, inplace=True)
                print(f"列 '{col}' 使用均值 {mean_val:.4f} 填充")
        
        # 对分类型列使用众数填充
        cat_cols = self.processed_data.select_dtypes(include=['object']).columns.tolist()
        for col in cat_cols:
            if self.processed_data[col].isnull().sum() > 0:
                mode_val = self.processed_data[col].mode()[0]
                self.processed_data[col].fillna(mode_val, inplace=True)
                print(f"列 '{col}' 使用众数 '{mode_val}' 填充")
        
        print(f"\n缺失值处理后数据形状: {self.processed_data.shape}")
        print(f"剩余缺失值数量: {self.processed_data.isnull().sum().sum()}")
        
        return self.processed_data
    
    def create_labels(self):
        """创建多标签"""
        print("\n" + "=" * 60)
        print("步骤4: 创建多标签")
        print("=" * 60)
        
        print(f"标签阈值: {LABEL_THRESHOLD * 100}%")
        
        # 创建二值标签
        self.processed_data['Label_A'] = (self.processed_data['A得分率'] >= LABEL_THRESHOLD).astype(int)
        self.processed_data['Label_B'] = (self.processed_data['B得分率'] >= LABEL_THRESHOLD).astype(int)
        self.processed_data['Label_C'] = (self.processed_data['C得分率'] >= LABEL_THRESHOLD).astype(int)
        
        # 统计标签分布
        label_stats = {
            'Label_A': self.processed_data['Label_A'].sum(),
            'Label_B': self.processed_data['Label_B'].sum(),
            'Label_C': self.processed_data['Label_C'].sum()
        }
        
        print("\n标签分布统计:")
        for label, count in label_stats.items():
            print(f"  {label}: {count} 个样本 ({count/len(self.processed_data)*100:.1f}%)")
        
        # 多标签组合统计
        self.processed_data['label_combo'] = (
            self.processed_data['Label_A'].astype(str) + 
            self.processed_data['Label_B'].astype(str) + 
            self.processed_data['Label_C'].astype(str)
        )
        combo_counts = self.processed_data['label_combo'].value_counts()
        
        print("\n标签组合分布:")
        for combo, count in combo_counts.items():
            labels = []
            if combo[0] == '1': labels.append('A')
            if combo[1] == '1': labels.append('B')
            if combo[2] == '1': labels.append('C')
            label_str = '+'.join(labels) if labels else '无标签'
            print(f"  {label_str}: {count} 个样本")
        
        self.preprocessing_report['标签分布'] = label_stats
        self.preprocessing_report['标签组合分布'] = combo_counts.to_dict()
        
        return self.processed_data
    
    def identify_feature_columns(self):
        """识别特征列"""
        print("\n" + "=" * 60)
        print("步骤5: 识别特征列")
        print("=" * 60)
        
        # 获取所有列
        all_cols = self.processed_data.columns.tolist()
        
        # 排除非特征列
        exclude_cols = CATEGORICAL_COLS + LABEL_COLS + ['Label_A', 'Label_B', 'Label_C', 'label_combo']
        
        # 识别可用的品质指标列
        self.feature_cols = []
        for col in ALL_QUALITY_COLS:
            if col in all_cols and col not in exclude_cols:
                self.feature_cols.append(col)
        
        print(f"识别到的特征列数量: {len(self.feature_cols)}")
        print(f"特征列: {self.feature_cols}")
        
        self.preprocessing_report['特征列数量'] = len(self.feature_cols)
        self.preprocessing_report['特征列'] = self.feature_cols
        
        return self.feature_cols
    
    def standardize_features(self):
        """特征标准化"""
        print("\n" + "=" * 60)
        print("步骤6: 特征标准化")
        print("=" * 60)
        
        # 提取特征数据
        X = self.processed_data[self.feature_cols].copy()
        
        # 标准化前统计
        print("\n标准化前特征统计:")
        pre_stats = X.describe().T[['mean', 'std', 'min', 'max']]
        print(pre_stats.head(10))
        
        # 执行标准化
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=self.feature_cols,
            index=X.index
        )
        
        # 更新处理后的数据
        for col in self.feature_cols:
            self.processed_data[col] = X_scaled[col]
        
        # 标准化后统计
        print("\n标准化后特征统计:")
        post_stats = X_scaled.describe().T[['mean', 'std', 'min', 'max']]
        print(post_stats.head(10))
        
        self.preprocessing_report['标准化前统计'] = pre_stats.to_dict()
        self.preprocessing_report['标准化后统计'] = post_stats.to_dict()
        
        return self.processed_data
    
    def split_dataset(self):
        """划分数据集"""
        print("\n" + "=" * 60)
        print("步骤7: 数据集划分")
        print("=" * 60)
        
        # 准备特征和标签
        X = self.processed_data[self.feature_cols]
        y = self.processed_data[['Label_A', 'Label_B', 'Label_C']]
        
        # 第一次划分: 训练集 vs (验证集+测试集)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=(VAL_RATIO + TEST_RATIO),
            random_state=RANDOM_STATE,
            stratify=y['Label_A']  # 使用主标签进行分层
        )
        
        # 第二次划分: 验证集 vs 测试集
        val_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=val_test_ratio,
            random_state=RANDOM_STATE
        )
        
        # 为了提升结果可复现性，对特征和标签子集按索引排序
        # 确保特征和标签一一对应
        train_idx = X_train.index.sort_values()
        val_idx = X_val.index.sort_values()
        test_idx = X_test.index.sort_values()
        
        X_train = X_train.loc[train_idx]
        y_train = y_train.loc[train_idx]
        X_val = X_val.loc[val_idx]
        y_val = y_val.loc[val_idx]
        X_test = X_test.loc[test_idx]
        y_test = y_test.loc[test_idx]
        
        print(f"数据集划分比例: 训练集:验证集:测试集 = {TRAIN_RATIO}:{VAL_RATIO}:{TEST_RATIO}")
        print(f"\n训练集大小: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"验证集大小: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
        print(f"测试集大小: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        
        # 各数据集标签分布
        print("\n各数据集标签分布:")
        for name, y_data in [('训练集', y_train), ('验证集', y_val), ('测试集', y_test)]:
            print(f"\n{name}:")
            for col in ['Label_A', 'Label_B', 'Label_C']:
                count = y_data[col].sum()
                print(f"  {col}: {count} ({count/len(y_data)*100:.1f}%)")
        
        self.preprocessing_report['数据集划分'] = {
            '训练集': len(X_train),
            '验证集': len(X_val),
            '测试集': len(X_test)
        }
        
        return (X_train, X_val, X_test, y_train, y_val, y_test)
    
    def run_preprocessing(self):
        """执行完整的预处理流程"""
        print("\n" + "=" * 80)
        print("                    数据预处理流程开始")
        print("=" * 80)
        
        self.load_data()
        self.analyze_missing_values()
        self.handle_missing_values()
        self.create_labels()
        self.identify_feature_columns()
        self.standardize_features()
        datasets = self.split_dataset()
        
        print("\n" + "=" * 80)
        print("                    数据预处理完成")
        print("=" * 80)
        
        return datasets, self.feature_cols, self.scaler, self.preprocessing_report


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    datasets, feature_cols, scaler, report = preprocessor.run_preprocessing()
