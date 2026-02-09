# -*- coding: utf-8 -*-
"""
烟叶风味多标签分类模型 - 配置文件
"""

import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# 确保目录存在
for dir_path in [DATA_DIR, OUTPUT_DIR, MODEL_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 数据文件
DATA_FILE = os.path.join(DATA_DIR, 'peifutf8.csv')

# 标签阈值
LABEL_THRESHOLD = 0.5

# 数据集划分比例 (训练:验证:测试 = 7:2:1)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# 随机种子
RANDOM_STATE = 42

# 字段分类配置
CATEGORICAL_COLS = ['样品编号', '省份', '品种', '等级', '部位', '香型', '样品类别']
LABEL_COLS = ['A得分率', 'B得分率', 'C得分率']

# 品质指标分类
SENSORY_COLS = ['香型风格', '彰显程度', '香气质', '香气量', '透发性', '杂气', 
                '刺激性', '余味', '甜感', '浓度', '细腻程度', '劲头', 
                '可用性赋分', '烟气特征', '香气特征', '口感特征']

CHEMICAL_COLS = ['总植物碱', '总氮', '还原糖', '总糖', '钾', '氯', '淀粉']

PHYSICAL_COLS = ['厚度', '密度', '单叶重', '平衡含水率', '拉力', '伸长量', '填充值', '含梗率']

APPEARANCE_COLS = ['叶片结构', '成熟度', '油分', '色度', '身份', '颜色']

# 所有品质指标
ALL_QUALITY_COLS = SENSORY_COLS + CHEMICAL_COLS + PHYSICAL_COLS + APPEARANCE_COLS

# 互信息阈值
# 说明：该阈值用于筛选与目标标签具有显著相关性的特征
# 选取依据：基于经验值，0.05表示特征与标签之间存在弱相关性以上的关系
# 低于此阈值的特征被认为对预测贡献较小，将被移除
MI_THRESHOLD = 0.05

# RFE特征筛选数量配置
# 说明：递归特征消除(RFE)筛选后保留的特征数量
# 设置为None时，自动计算为 max(5, 总特征数 // 2)
# 可根据实际需求调整此参数
RFE_N_FEATURES_TO_SELECT = None

# 随机森林参数搜索空间
RF_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# 图表配置
FIGURE_DPI = 150
FIGURE_SIZE = (12, 8)
