# Publaynet
+ [Publaynet: Largest Dataset Ever for Document Layout Analysis](https://arxiv.org/pdf/1908.07836)

## Requirements:
[Detectron2](https://github.com/facebookresearch/detectron2)
- cython
- torchvision
- opencv-python
- 'git+https://github.com/facebookresearch/fvcore'
- 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

## DataSet:
- Images
  * [001-100](https://bhpan.buaa.edu.cn:443/link/4399929A767FFDB1050AF5B5BA055073)
  * [101-200](https://bhpan.buaa.edu.cn:443/link/9F28152E98CF60E531195B8E6640EF2C)
  * [201-300](https://bhpan.buaa.edu.cn:443/link/877D5DAC0B19BFAE6AFFA97D92B14477)
  * [301-400](https://bhpan.buaa.edu.cn:443/link/E142647428D4D3E18544D865B944A87F)
  * [401-500](https://bhpan.buaa.edu.cn:443/link/D6D4B32C95E41C2D374981A2C43B7827)
  
- Annotation
  * [Txt](https://bhpan.buaa.edu.cn:443/link/0E4FDB66D538F60A891E51CBB94F09A7)
  * [Json](https://bhpan.buaa.edu.cn:443/link/B1934FD5815D3F3F89323239CEBC73B3)

## To Run in Google Colab:
1. Clone publaynet and detectron2
```
!git clone https://github.com/noba1anc3/Publaynet.git
cd Publaynet
!git clone https://github.com/facebookresearch/detectron2.git
```

2. Install Requirements and Build Detectron2
```
pip install -r requirements.txt
cd detectron2
!python setup.py build develop
cd ..
```

3. Import Google Drive
```
from google.colab import drive
drive.mount('/content/drive/')
```

4. Copy data and model to Publaynet's path
```
mkdir publaynet_data
mkdir output_publaynet
cp -rf ../drive/'My Drive'/image/ ./publaynet_data/
cp -rf ../drive/'My Drive'/data.json ./publaynet_data/
cp -rf ../drive/'My Drive'/model_final.pth ./output_publaynet
```

## 文件说明：
* detectron2_repo: 需要下载的Detectron包，见GitHub下载指导
* faster_rcnn_resnet101_coco_2018_01_28: backbone的预训练模型，用于publaynet数据集训练
* output_publaynet: 使用backbone的预训练模型在publaynet上训练的结果
* output: 使用publaynet作为预训练模型在自己的数据集上训练的结果
* build.py: 构建优化器和学习率策略
* predict.py: 使用publaynet数据集的预测文件
* train.py: 使用publaynet数据集的训练文件
* utils.py: 使用publaynet数据集的工具文件
* test_per_img.py: 可视化测试集的预测结果
* visualizeSet.py: 可视化数据集
