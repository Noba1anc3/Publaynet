# Publaynet项目结构说明

*****
## 安装环境：
### Detectron2 
* Linux or macOS with Python ≥ 3.6
* PyTorch ≥ 1.3
* pycocotools
* json
* [Github地址](https://github.com/facebookresearch/detectron2)
* [Colab教程](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)

*****
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
