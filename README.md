# 一、项目简介
本项目应用于IDC机房UPS电池故障预测，首选预测未来一段时间的电压（或其他判断电池故障的指标，如电池内阻），然后再根据业务人员判断是否故障的指标阈值进而判断每个时间点是否为异常值，即该电池在未来哪一个时间点或时间段可能会发生故障。
# 二、目录架构
**meta-42 battery failure prediction**

```python
# 存放训练集测试集
├── data 
│   ├── data_process.py   # 数据处理
│   ├── m6_test			  # 测试集
│   └── m6_train		  # 训练集

# 存放训练的模型文件
├── model_checkpoints

# 存放主要模型代码
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

# 运行程序的文件
├── run_examples
│   ├── main.py
│   ├── my_main.py  # 运行程序入口
```

# 三、安装依赖
> `pip install -r requirements.txt`

# 四、使用教程及细节
> `python run_examples/my_main.py  --data 'data_path'  --kpi_list  ['InternalVol']  --num_obs_to_train 60 --predict_seq_len 10 --min_thre 12.6 --max_thre 15.0`
    
> 参数含义依次为：数据路径、使用的特征指标字段、训练的子时间序列长度、预测的时间长度、正常阈值最小值、正常阈值最大值。

本项目主要针对多维特征的时间序列数据进行预测，
**输入的训练集：** 其shape为（n, m）的时间序列数据，其中n为设定的训练时间序列长度(窗口)，m为特征向量的维度(如电压、内阻等)；
**输入训练集对应的标签：** 其shape为 ($n'$, $m'$)，其中$n' = n+t$,  $t$ 为 n后续连续的 $t$个时间序列单位，即预测未来多少个时间序列单位；
**输出：** 和输入的标签一样的shape（$n'$, $m'$)）；
**判断逻辑：** 对预测的值进行阈值判断，超出阈值范围则当前时间序列点为故障点，（如正常电压的阈值范围为：12.6V—15.0V，超出这个阈值范围则为故障。）

# 五、参考
https://github.com/comp5331-Xtimeseries/TPA-LSTM-NotOrigional
https://github.com/Sanyam-Mehta/TPA-LSTM-PyTorch
https://github.com/shunyaoshih/TPA-LSTM

