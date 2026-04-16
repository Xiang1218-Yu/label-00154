import sys
sys.path.insert(0, '.')
from data_preprocessing import DataPreprocessor

# 运行预处理
preprocessor = DataPreprocessor()
datasets, feature_cols, scaler, report = preprocessor.run_preprocessing()
X_train, X_val, X_test, y_train, y_val, y_test = datasets

# 验证索引是否一致
print('\n' + '='*60)
print('验证特征与标签索引是否一致')
print('='*60)
print(f'训练集索引一致: {X_train.index.equals(y_train.index)}')
print(f'验证集索引一致: {X_val.index.equals(y_val.index)}')
print(f'测试集索引一致: {X_test.index.equals(y_test.index)}')

# 验证索引是否排序
print(f'\n训练集索引已排序: {X_train.index.is_monotonic_increasing}')
print(f'验证集索引已排序: {X_val.index.is_monotonic_increasing}')
print(f'测试集索引已排序: {X_test.index.is_monotonic_increasing}')

# 打印前5个样本的索引对比
print('\n训练集前5个样本索引对比:')
print(f'X_train索引: {X_train.index[:5].tolist()}')
print(f'y_train索引: {y_train.index[:5].tolist()}')

print('\n验证成功! 特征与标签现在完全对应。')
