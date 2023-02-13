# OpenBG-IMG Benchmark

OpenBG-IMG is a multi-modal dataset in the field of e-commerce. This benchmark implements a variety of knowledge graph embedding models for link prediction, which is used to generate evaluation results of [CCKS2022 Knowledge Processing and Application Evaluation for Digital Commerce Task 3: Multimodal Commodity Knowledge Graph Link Prediction](https://tianchi.aliyun.com/competition/entrance/531957/introduction). Please submit the results to TIANCHI platform.

# Requirements

Use the following code to configure the environment.
```
pip install -r requirements.txt
```

# Dataset

- [https://drive.google.com/file/d/1jg4YcFgOfgjUJCnxBjw9w-6ID8VS_L-X/view?usp=sharing](https://drive.google.com/file/d/1jg4YcFgOfgjUJCnxBjw9w-6ID8VS_L-X/view?usp=sharing)

Please download the dataset from TIANCHI to the directory `./data/`.

```
data
 |-- OpenBG-IMG
 |    |-- images            # Set of images
 |    |    |-- ent_xxxxxx   # Images of the entity
 |    |    |-- ...
 |    |-- train.tsv         # Training set
 |    |-- test.tsv          # Test set
```

The statistics of OpenBG-IMG：
|    Dataset    |    # Ent   | # Rel |   # Train   |  # Dev  | # Test  |
| ------------- | ---------- | ----- | ----------- | ------- | ------- |
|   OpenBG-IMG  | 27,910†     |  136  | 230,087     | 5,000   | 14,675  |

†: there are 14,718 multi-modal entities in OpenBG-IMG.

#### Check the data

```
$ head -n 3 train.tsv
ent_021198	rel_0031	ent_017656
ent_008185	rel_0092	ent_025949
ent_005940	rel_0080	ent_020805
```

# Quick Start


## TransE & TransH & TransE & DistMult & ComplEx

These models refer to the implement of [OpenKE](https://github.com/thunlp/OpenKE).

- Compile C++ files

```shell
    cd [The root directory of model]
    bash scripts/make.sh
```

- Preprocess data

```shell
    bash scripts/prepro.sh
```

- Train model and predict results saved to `./results/result.tsv`


```shell
    bash scripts/train.sh
```

## TuckER

The model refers to the implement of [TuckER](https://github.com/ibalazevic/TuckER).

- Preprocess data

```shell
    bash scripts/prepro.sh
```

- Train model and predict results saved to `./results/result.tsv`


```shell
    bash scripts/train.sh
```

## TransAE

The model refers to the implement of the TransE model in [OpenKE](https://github.com/thunlp/OpenKE) and the process of images in [TransAE](https://github.com/ksolaiman/TransAE).

- Compile C++ files

```shell
    cd TransAE
    bash scripts/make.sh
```

- Preprocess data

```shell
    bash scripts/prepro.sh
```

- Get representations and encodings of images


```shell
    bash scripts/visual_emb.sh
```

- Train model and predict results saved to `./results/result.tsv`


```shell
    bash scripts/train.sh
```

## RSME

The model refers to the implement of [RSME](https://github.com/wangmengsd/RSME).

- Get representations and encodings of images

```shell
    cd RSME
    bash scripts/visual_emb.sh
```

- Preprocess data

```shell
    bash scripts/prepro.sh
```

- Train model and predict results saved to `./results/result.tsv`

```shell
    bash scripts/train.sh
```

# Experiments

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

# Acknowledgements

Thanks for the following works!

- [https://github.com/thunlp/OpenKE](https://github.com/thunlp/OpenKE)
- [https://github.com/ibalazevic/TuckER](https://github.com/ibalazevic/TuckER)
- [https://github.com/ksolaiman/TransAE](https://github.com/ksolaiman/TransAE)
- [https://github.com/wangmengsd/RSME](https://github.com/wangmengsd/RSME)

# More related works

Multi-modal Knowledge Graph completion: MKGFormer([https://github.com/zjunlp/MKGformer/](https://github.com/zjunlp/MKGformer/))
