## 全流程预估-TensorFlow2.0版本

### 一、路径说明

- 特征配置：config/feature_config/feature_infos.py
- DNN全局配置：config/global_config.py
- TextCNN全局配置：config/global_config_text.py



### 二、运行方式

```python
source activate py3
cd ~/whole_process_predict-tensorflow
```



- 运行TextCNN：python text_exp/train.py
- 运行DNN：python model/DNN.py
- 运行DFM：python model/DFM.py

## 三、best实验结果

训练：20190725-20190813

测试：20190821-20190823

### 只用微信文本

- XGB：0.868
- TextCNN:0.835

### 只用基本特征

- XGB：0.852
- DNN：0.78
- DFM：**