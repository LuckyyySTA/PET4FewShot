# PET4FewShot
[PET](https://arxiv.org/abs/2001.07676) (Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference) 提出将输入示例转换为完形填空式短语，以帮助语言模型理解给定的任务

## 代码结构及说明

```
|—— FewCLUE	# FewCLUE数据集文件夹
	|——datasets	# 9个数据集文件夹
		|——tnews	# tnew数据集文件夹
		|——...
|—— label_normalized	# 储存了PET对标签进行转化的json文件
|—— pet.py # PET 策略的训练、评估主脚本
|—— data.py # PET 策略针对 FewCLUE 9 个数据集的任务转换逻辑，以及明文 -> 训练数据的转换
|—— evaluate.py # 针对 FewCLUE 9 个数据集的评估函数
|—— predict.py # 针对 FewCLUE 9 个数据集进行预测
```

**tips:** 目前只完成了tnews数据集，剩余8个数据集to be done

## 基于FewCLUE进行PET实验

#### 数据准备

数据集均存放在`FewCLUE/datasets`文件夹下

#### 模型训练

运行`sh run.sh`脚本在指定数据集上进行训练&评估

参数含义说明

- `task_name`: FewCLUE 中的数据集名字
- `device`: 使用 cpu/gpu 进行训练
- `pattern_id` 完形填空的模式
- `save_dir`: 模型存储路径
- `max_seq_length`: 文本的最大截断长度

模型每训练 1 个 epoch, 会在验证集上进行评估

#### 模型预测

运行`sh predict.sh`脚本在指定数据集上进行预测

## References

[1] Schick, Timo, and Hinrich Schütze. “Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference.” ArXiv:2001.07676 [Cs], January 25, 2021. http://arxiv.org/abs/2001.07676.

[2]FewCLUE数据集：https://github.com/CLUEbenchmark/FewCLUE

[3]基于PaddlePaddle的官方实现：https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/few_shot/pet

