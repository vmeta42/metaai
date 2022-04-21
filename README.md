# 一、项目简介
本项目应用于IDC机房UPS电池故障预测，首选预测未来一段时间的电压（或其他判断电池故障的指标，如电池内阻），然后再根据业务人员判断是否故障的指标阈值进而判断每个时间点是否为异常值，即该电池在未来哪一个时间点或时间段可能会发生故障。
# 二、目录架构
**meta-42 battery failure prediction**

```python
# 存放训练集测试集
├── data 
│   ├── data_process.py   # 数据处理
│   ├── m6_test			  # 存放my_tsp测试集
│   └── m6_train		  # 存放my_tsp训练集
│  	└── pm_datasets		  # 存放my_pm数据集

# 存放训练的模型文件
├─model_checkpoints
│  ├─pm_checkpoints       # 存放my_pm训练好的模型文件   
│  └─tsp_checkpoints	  # 存放my_tsp训练好的模型文件

# 存放主要模型代码
# my_pm 项目主要模型代码
├─my_pm
│  │  my_test.py  		  # 测试代码 
│  │  my_trainer.py		  # 训练代码
│  │  pm_main.py		  # 主运行代码
│  ├─configs			  # 存放模型配置文件
│  │      lightgbm.yaml
│  │      xgboost.yaml
│  │      
│  ├─datasets			    # 处理数据的文件
│  │  │  auto_labeling.py	# 自动标注模块
│  │  │  data_processing.py	# 数据预处理
│  │  │  pm_dataset.py		# 构建数据
│  │          
│  ├─evaluation_metrics     # 模型评估指标方法，如果有特殊的应用场景需要按照需求写
│  │          
│  ├─models                 # 存放模型代码
│  │  │  cls_model.py		# 模型代码 
│  │              
│  ├─utils					# 存放一些工具函数
│  │  │  util.py			# 有自动创建文件等工具函数


├── my_tsp
│   ├── __init__.py
│   ├── datasets  # 存放处理数据程序文件
│   │   ├── __init__.py
│   │   ├── data_processing.py  # 数据处理，如各种的数据归一化、标准化等
│   │   └── ts_dataset.py		# 生成dataloader，便于分批次读取数据训练和测试
│   ├── evaluation_metrics  # 存放评估方法和损失函数
│   │   ├── __init__.py
│   │   ├── evaluate.py  # 根据业务场景设定评估指标，包括分类评估指标(混淆矩阵, P, R，F1-score等)
│   │   └── loss.py		 # 损失函数，包括分类和回归损失函数
│   ├── models    # 存放模型程序的文件
│   │   ├── LSTM.py  # 时间序列模型LSTM，在输出后面使用了cov卷积，使得预测的时间序列长度和指标的维数是可调的
│   │   ├── __init__.py
│   │   ├── my_tpaLSTM.py	# LSTM + attention
│   │   └── tpaLSTM.py
│   ├── my_trainer.py  # 调用数据，调用模型，训练模型
│   ├── test.py  # 测试模型，输出评估结果
│   ├── trainer.py
│   └── utils  # 一些工具函数，如计算最大斜率差、预测的回归值转换为类别标签、画预测曲线图等工具
│       ├── __init__.py
│       └── util.py

# 存放测试集预测结果曲线图，方便可视化评估 
├── outputs_pics

# my_tsp运行程序的文件
├── run_examples
│   ├── main.py
│   ├── my_main.py  # 运行程序入口
```

# 三、安装依赖
> `pip install -r requirements.txt`

# 四、代码结构简介以及教程
本项目主要使用两种思路来解决电池故障预测，一种是基于深度学习方法，一种是基于传统机器学习方法。
**基于深度学习的方法：** 代码在my_tsp文件夹中。其故障预测的思路为：将其看作是时间序列问题，输入多维特征的时间序列数据到LSTM模型（或其变种模型）中，预测未来若干时间段的电压、内阻等值，然后根据判断电池是否故障的逻辑——超出电压、内阻的正常阈值，对电池故障进行预测。
> `python run_examples/my_main.py  --data 'data_path'  --kpi_list  ['InternalVol']  --num_obs_to_train 60 --predict_seq_len 10 --min_thre 12.6 --max_thre 15.0`

> 参数含义依次为：数据路径、使用的特征指标字段、训练的子时间序列长度、预测的时间长度、正常阈值最小值、正常阈值最大值。

本项目主要针对多维特征的时间序列数据进行预测，
**输入的训练集：** 其shape为（n, m）的时间序列数据，其中n为设定的训练时间序列长度(窗口)，m为特征向量的维度(如电压、内阻等)；
**输入训练集对应的标签：** 其shape为 (n, m)，其中n = n+t,  t 为 n后续连续的 t个时间序列单位，即预测未来多少个时间序列单位；
**输出：** 和输入的标签一样的shape（n, m)）；
**判断逻辑：** 对预测的值进行阈值判断，超出阈值范围则当前时间序列点为故障点，（如正常电压的阈值范围为：12.6V—15.0V，超出这个阈值范围则为故障。）

**基于机器学习的方法：**  代码在my_pm文件夹中。其故障预测的思路为：根据业务人员对电池故障预测的判断逻辑，并对大量的数据曲线进行分析，然后对数据进行自动标注，标注的故障点(电池更换时间点)为真实故障点前一个月左右的时间(自动标注中的算法中实现)，然后将其作为分类问题，使用GBDT、xgboost lightGBM等算法进行分类。本模块是对**[此论文](https://www.sciencedirect.com/science/article/pii/S240589632031185X)**的一个简单复现。
> python  my_pm/pm_main.py --csv_files_path 'csv原始数据文件路径' 
ps： 初次运行为加速每次训练，将csv文件经过数据处理后转换为 train_cache.npy 和 test_cache.npy, 之后训练测试加载这两个数据方便加速训练。
# 五、参考

https://github.com/comp5331-Xtimeseries/TPA-LSTM-NotOrigional
https://github.com/Sanyam-Mehta/TPA-LSTM-PyTorch
https://github.com/shunyaoshih/TPA-LSTM
https://www.sciencedirect.com/science/article/pii/S240589632031185X

