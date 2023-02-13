# OpenBG-IMG基准

<p align="left">
    <b> 简体中文 | <a href="https://github.com/OpenBGBenchmark/OpenBG-IMG/blob/main/README_EN.md">English</a> </b>
</p>

OpenBG-IMG是电子商务领域的多模态数据集，本基准使用了多种知识图谱嵌入模型进行链接预测，用于生成[CCKS2022面向数字商务的知识处理与应用评测任务三：多模态商品知识图谱链接预测](https://tianchi.aliyun.com/competition/entrance/531957/introduction)的评测结果，评测结果请在阿里天池平台进行提交。

## 环境配置

使用以下代码进行环境配置
```
pip install -r requirements.txt
```

## 数据集

- [https://drive.google.com/file/d/1jg4YcFgOfgjUJCnxBjw9w-6ID8VS_L-X/view?usp=sharing](https://drive.google.com/file/d/1jg4YcFgOfgjUJCnxBjw9w-6ID8VS_L-X/view?usp=sharing)

请将天池平台上的数据放置在`./data/`，数据目录如下

```
data
 |-- OpenBG-IMG
 |    |-- images            # 图片集
 |    |    |-- ent_xxxxxx   # 实体对应图片
 |    |    |-- ...
 |    |-- train.tsv         # 训练集数据
 |    |-- test.tsv          # 测试集数据
```

数据集统计数据如下：
|    Dataset    |    # Ent   | # Rel |   # Train   |  # Dev  | # Test  |
| ------------- | ---------- | ----- | ----------- | ------- | ------- |
|   OpenBG-IMG  | 27,910†     |  136  | 230,087     | 5,000   | 14,675  |

†:实体中有14,718个多模态实体

#### 查看数据集数据

```
$ head -n 3 train.tsv
ent_021198	rel_0031	ent_017656
ent_008185	rel_0092	ent_025949
ent_005940	rel_0080	ent_020805
```

## 如何运行


### TransE & TransH & TransE & DistMult & ComplEx

模型参考并修改了[OpenKE](https://github.com/thunlp/OpenKE)中的实现。

- 编译C++代码

```shell
    cd 模型目录
    bash scripts/make.sh
```

- 数据预处理

```shell
    bash scripts/prepro.sh
```

- 训练模型并预测结果，结果保存在`./results/result.tsv`


```shell
    bash scripts/train.sh
```

### TuckER

模型参考并修改了[TuckER](https://github.com/ibalazevic/TuckER)中的实现。

- 数据预处理

```shell
    bash scripts/prepro.sh
```

- 训练模型并预测结果，结果保存在`./results/result.tsv`


```shell
    bash scripts/train.sh
```

### TransAE

本模型参考了[OpenKE](https://github.com/thunlp/OpenKE)中TransE模型的实现以及[TransAE](https://github.com/ksolaiman/TransAE)中对图片的表示和编码。

- 编译C++代码

```shell
    cd TransAE
    bash scripts/make.sh
```

- 数据预处理

```shell
    bash scripts/prepro.sh
```

- 获取图片表示和编码


```shell
    bash scripts/visual_emb.sh
```

- 训练模型并预测结果，结果保存在`./results/result.tsv`


```shell
    bash scripts/train.sh
```

### RSME

本模型参考了[RSME](https://github.com/wangmengsd/RSME)的官方代码。

- 获取图片表示和编码

```shell
    cd RSME
    bash scripts/visual_emb.sh
```

- 数据预处理

```shell
    bash scripts/prepro.sh
```

- 训练模型并预测结果，结果保存在`./results/result.tsv`

```shell
    bash scripts/train.sh
```

## 实验结果

|Model		|HIT@1  |HIT@3 |HIT@10| MR  | MRR |
|:-		|:-:	|:-:   |:-:   |:-:  |:-:  |
|TransE	    |0.150  |0.387 |0.647 |118  |0.315|
|TransH 	|0.129  |0.525 |0.743 |112  |0.357|
|TransD	    |0.137  |0.532 |0.746 |110  |0.364|
|DistMult	|0.060  |0.157 |0.279 |524  |0.139|
|ComplEx	|0.143  |0.244 |0.371 |782  |0.221|
|TuckER	    |0.497  |0.690 |0.820 |1473 |0.611|
|TransAE	|0.274  |0.489 |0.715 |36.1 |0.421|
|RSME       |0.485  |0.687 |0.838 |72.1 |0.607|

## 致谢

此代码参考了以下代码：

- [https://github.com/thunlp/OpenKE](https://github.com/thunlp/OpenKE)
- [https://github.com/ibalazevic/TuckER](https://github.com/ibalazevic/TuckER)
- [https://github.com/ksolaiman/TransAE](https://github.com/ksolaiman/TransAE)
- [https://github.com/wangmengsd/RSME](https://github.com/wangmengsd/RSME)

十分感谢！

## 更多相关工作

关于多模态知识图谱构建和补全的开源工作请参见MKGFormer([https://github.com/zjunlp/MKGformer/](https://github.com/zjunlp/MKGformer/))
