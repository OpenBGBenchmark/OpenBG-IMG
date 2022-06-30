# OpenBG-IMG基准

OpenBG-IMG是电子商务领域的多模态数据集，本基准使用了多种知识图谱嵌入模型进行链接预测，用于生成[CCKS2022面向数字商务的知识处理与应用评测任务三：多模态商品知识图谱链接预测](https://tianchi.aliyun.com/competition/entrance/531957/introduction)的评测基准，评测结果请在阿里天池平台进行提交。

# 环境配置

使用以下代码进行环境配置
```
pip install -r requirements.txt
```

# 数据集格式

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

# 如何运行


## TransE & TransH & TransE & DistMult & ComplEx

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

## TransAE

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

## RSME

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

# 致谢

此代码参考了以下代码：

- [https://github.com/thunlp/OpenKE](https://github.com/thunlp/OpenKE)
- [https://github.com/ksolaiman/TransAE](https://github.com/ksolaiman/TransAE)
- [https://github.com/wangmengsd/RSME](https://github.com/wangmengsd/RSME)

十分感谢！

# 更多相关工作

关于多模态知识图谱构建和补全的开源工作请参见MKGFormer([https://github.com/zjunlp/MKGformer/](https://github.com/zjunlp/MKGformer/))
