# Publaynet
+ [Publaynet: Largest Dataset Ever for Document Layout Analysis](https://arxiv.org/pdf/1908.07836)

## Requirements:
- Detectron2 
  * [Github地址](https://github.com/facebookresearch/detectron2)
  * [Colab教程](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)

## DataSet:
- Images
  * [1-100](https://bhpan.buaa.edu.cn:443/link/62EE057BFDEE0EF4FCCB5B45297E79AB)
  * [101-200](https://bhpan.buaa.edu.cn:443/link/98152587BB8EFEB41559A583BFA57DDF)
  * [201-300](https://bhpan.buaa.edu.cn:443/link/9772C7FFA309BF3F230E54940F143DFD)
  * [301-400](https://bhpan.buaa.edu.cn:443/link/DA9CD07C69A956BD305A5B3A1627C91B)
  * [401-500](https://bhpan.buaa.edu.cn:443/link/03DF67362C1018773399AAE2183F1DDA)
  
- Annotation
  * [Txt](https://bhpan.buaa.edu.cn:443/link/4B533139455EA1148CEA19F7AEEB993F2)
  * [Json]()
 
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
