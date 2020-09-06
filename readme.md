# FAST DETECTION NET PROJECT

## 介绍

* 使用pytorch(1.5.1)实现通用的网络编写和训练/推理
* 使用tensorflow(2.2.0), keras(2.3.0-tf)实现通用的网络编写和训练/推理

### 1. mobilenet的实现

## 使用方法

1. 下载fast detection工程：

```bash
    git clone https://github.com/sumagic/fast-detection.git
```

2. 进入工程目录，设置环境变量

```bash
    cd fast-detection
    source set-env.sh
```

3. 分别下载数据集和第三方库

```bash
    cd ${FAST_DETECTION_ROOT}/3rd
    ./get_3rd.sh
    cd ${FAST_DETECTION_ROOT}/dataset
    ./get_dataset.sh
```

4. 执行build.sh

```bash
    ./build.sh
```

5. 运行用例

```bash
    cd ${FAST_DETECTION_ROOT}/sample
    ./xxx.py
```