# Publaynet
+ [Publaynet: Largest Dataset Ever for Document Layout Analysis](https://arxiv.org/pdf/1908.07836)

## Requirements
[Detectron2](https://github.com/facebookresearch/detectron2)
- Linux or macOS with Python ≥ 3.6
- cython
- opencv-python
- torchvision　(PyTorch ≥ 1.3)
- 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

## DataSet
### Overview
|  Category     | Training Set  | Validating Set  | Testing Set  |
|:-------------:|:-------------:|:---------------:|:------------:|
| Num of Images | 20365         | 500             | 499          |
| Percentage    | 95%           | 2.5%            | 2.5%         |

Training Set:

Validating Set:

| category | #instances | category | #instances | category | #instances | category | #instances |
|:--------:|:----------:|:--------:|:----------:|:--------:|:----------:|:--------:|:----------:|
| chapter  | 342        | section  | 561        | clause   | 3624       | total    | 4527       |

Testing Set:

| category | #instances | category | #instances | category | #instances | category | #instances |
|:--------:|:----------:|:--------:|:----------:|:--------:|:----------:|:--------:|:----------:|
| chapter  | 332        | section  | 549        | clause   | 3352       | total    | 4233       |

### Download
#### All Files:
Images
* [001-100](https://bhpan.buaa.edu.cn:443/link/4399929A767FFDB1050AF5B5BA055073)
* [101-200](https://bhpan.buaa.edu.cn:443/link/9F28152E98CF60E531195B8E6640EF2C)
* [201-300](https://bhpan.buaa.edu.cn:443/link/877D5DAC0B19BFAE6AFFA97D92B14477)
* [301-400](https://bhpan.buaa.edu.cn:443/link/E142647428D4D3E18544D865B944A87F)
* [401-500](https://bhpan.buaa.edu.cn:443/link/D6D4B32C95E41C2D374981A2C43B7827)

Annotation
* [Txt](https://bhpan.buaa.edu.cn:443/link/0E4FDB66D538F60A891E51CBB94F09A7)
* [Json](https://bhpan.buaa.edu.cn:443/link/B1934FD5815D3F3F89323239CEBC73B3)

#### Dataset:
Beihang Pan:
- [Training Set](https://bhpan.buaa.edu.cn/#/link/8652A7C4D9564589A017F078DF72D532?gns=6DB717ABC02F4A6794D661D007D50419%2FD3BB1FB487824A5BB26CE7A3F259D7B1%2F16F22C7FB23E4C8F80C5281445AAC8A3)
- [Validating Set](https://bhpan.buaa.edu.cn/#/link/8652A7C4D9564589A017F078DF72D532?gns=6DB717ABC02F4A6794D661D007D50419%2FD3BB1FB487824A5BB26CE7A3F259D7B1%2FCED866A3B19F451B85F6700804150471)
- [Testing Set](https://bhpan.buaa.edu.cn/#/link/8652A7C4D9564589A017F078DF72D532?gns=6DB717ABC02F4A6794D661D007D50419%2FD3BB1FB487824A5BB26CE7A3F259D7B1%2FF3CAF395CE5946758223D044616A894F)

Google Drive:
- [Training Set]()
- [Validating Set](https://drive.google.com/open?id=1U5EkjBXjIz9rFL74oNAojb3PsQFk5AGG)
- [Testing Set](https://drive.google.com/open?id=1or8Mzxp7Q-GOihxkfH7scCJIeNTs86ar)

## Model:
 - [Model finetuned with Publaynet Dataset based on pretrained model of Faster-RCNN-ResNet](https://drive.google.com/open?id=1DPfPmN7Z-aefzSCw_KcCPxi4ArTeG5cl)

## Python Files:
* faster_rcnn_resnet101_coco_2018_01_28: backbone的预训练模型，用于publaynet数据集训练
* visualizeSet.py: 可视化数据集
* build.py: 构建优化器和学习率策略
* utils.py: 使用publaynet数据集的工具文件
* train.py: 使用publaynet数据集的训练文件
* test_per_img.py: 可视化测试集的预测结果
* predict.py: 使用publaynet数据集的预测文件


## To Run on Google Colab:
### Install Requirements and Clone Publaynet
```
!pip install cython
!pip install torchvision
!pip install opencv-python
!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
!git clone https://github.com/noba1anc3/Publaynet.git
cd Publaynet
```

### Build Detectron2 from Source

After having the above dependencies and gcc & g++ ≥ 5, run:
```
!git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
!python -m pip install -e .
cd ..

# Or if you are on macOS
# CC=clang CXX=clang++ python -m pip install -e .
```

## Train
### Mount Google Drive
```
from google.colab import drive
drive.mount('/content/drive/')
```

### Copy Data and Model to Publaynet's Path
```
mkdir data
mkdir output

cp -rf ../drive/'My Drive'/val.zip ./data/
cp -rf ../drive/'My Drive'/test.zip ./data/
cp -rf ../drive/'My Drive'/model_final.pth ./output/

cd data
!unzip val.zip
!unzip test.zip
cd ..
```

### Start Training
```
!python train.py
```
