# 烟叶风味多标签分类模型

基于随机森林算法的烟叶风味多标签分类系统，通过品质指标预测烟叶的ABC三类风味标签。

## How to Run

### Docker启动（推荐）

1. 准备数据文件
```bash
# 将数据文件 peifutf8.csv 放置到 backend/data/ 目录下
mkdir -p backend/data
cp /path/to/peifutf8.csv backend/data/
```

2. 构建并运行
```bash
docker compose up --build -d
```

3. 查看运行日志
```bash
docker compose logs -f
```

4. 查看输出结果
```bash
# 分析报告和图表保存在 backend/output/ 目录
# 训练好的模型保存在 backend/models/ 目录
ls backend/output/
ls backend/models/
```

5. 停止服务
```bash
docker compose down
```

### 本地启动

1. 创建虚拟环境
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 准备数据
```bash
# 将数据文件 peifutf8.csv 放置到 backend/data/ 目录下
mkdir -p data
cp /path/to/peifutf8.csv data/
```

4. 运行程序
```bash
python main.py
```

## Services

| 服务名称 | 描述 | 端口 |
|---------|------|------|
| tobacco-flavor-model | 烟叶风味多标签分类模型 | 无（批处理任务） |

## 测试账号

本项目为数据分析批处理任务，无需登录账号。

## 题目内容

### 项目背景

邀请5名专业感官测评人员组成的评审组，评价了150个烟叶样品的风味特征。评审人员给烟叶贴上ABC三类风味标签，部分专家给烟叶贴了不止1个标签，产生了多标签评价数据。

### 数据说明

- **样本数量**: 150个烟叶样品
- **标签阈值**: 50%（当某个标签得分率超过50%时，认为该样品具备该风味）
- **数据集划分**: 训练集:验证集:测试集 = 7:2:1

### 品质指标分类

1. **评吸指标**: 香型风格、彰显程度、香气质、香气量、透发性、杂气、刺激性、余味、甜感、浓度、细腻程度、劲头、可用性赋分、烟气特征、香气特征、口感特征

2. **化学成分**: 总植物碱、总氮、还原糖、总糖、钾、氯、淀粉

3. **物理特性**: 厚度、密度、单叶重、平衡含水率、拉力、伸长量、填充值、含梗率

4. **外观质量**: 叶片结构、成熟度、油分、色度、身份、颜色

### 分析流程

1. **数据预处理**
   - 缺失值处理（均值/众数填充）
   - 数据标准化（Z-score标准化）

2. **特征工程**
   - 递归特征消除(RFE)初步筛选
   - 互信息值分析二次筛选
   - 特征交互项生成

3. **模型构建**
   - 随机森林多标签分类模型
   - 网格搜索超参数优化

4. **验证体系**
   - 混淆矩阵评估
   - 准确率、精确率、召回率、F1分数

5. **实施优化**
   - 模型保存与加载
   - 特征重要性排序

### 输出内容

- `output/analysis_report.md` - 详细分析报告
- `output/*.png` - 可视化图表
- `models/tobacco_flavor_model.pkl` - 训练好的模型
- `models/feature_engineering_params.pkl` - 特征工程参数

---

## 项目结构

```
.
├── backend/
│   ├── config.py              # 配置文件
│   ├── data_preprocessing.py  # 数据预处理模块
│   ├── feature_engineering.py # 特征工程模块
│   ├── model_training.py      # 模型训练模块
│   ├── visualization.py       # 可视化模块
│   ├── report_generator.py    # 报告生成模块
│   ├── main.py               # 主程序
│   ├── requirements.txt      # Python依赖
│   ├── Dockerfile            # Docker构建文件
│   ├── data/                 # 数据目录
│   ├── output/               # 输出目录
│   └── models/               # 模型目录
├── docker compose.yml        # Docker Compose配置
├── .gitignore               # Git忽略文件
└── README.md                # 项目说明
```

## 技术栈

- **语言**: Python 3.11
- **机器学习**: scikit-learn
- **数据处理**: pandas, numpy
- **可视化**: matplotlib, seaborn
- **容器化**: Docker

## 注意事项

1. Windows系统运行时，程序已配置中文字体支持
2. Docker镜像支持ARM64和AMD64架构
3. 如无数据文件，程序会自动生成示例数据用于演示
