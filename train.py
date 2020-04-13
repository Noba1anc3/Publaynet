
import os
import sys, getopt

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

def main(argv):

    setup_logger()

    finetune = False
    trainPath = "./data/train/images/"
    trainjsonPath = "./data/train/train.json"
    testPath = "./data/val/images/"
    testjsonPath = "./data/val/val.json"

    opts, args = getopt.getopt(argv, "hf:", ["finetune="])

    for opt, arg in opts:
        if opt == '-f':
            if arg == 'True':
                finetune = True

    register_coco_instances("trainSet", {}, trainjsonPath, trainPath)
    register_coco_instances("testSet", {}, testjsonPath, testPath)

    cfg = get_cfg()

    # cfg.MODEL.DEVICE = 'cpu'
    
    cfg.OUTPUT_DIR = './output'
    cfg.merge_from_file("./detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("trainSet",)
    cfg.DATASETS.TEST = ("testSet",)
    cfg.DATALOADER.NUM_WORKERS = 6
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 1e-3
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

    if finetune:
        cfg.SOLVER.MAX_ITER = 190500
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0005499.pth")
    else:
        cfg.SOLVER.MAX_ITER = 500
        cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    for i in range(16):
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=True)
        trainer.train()
        evaluator = COCOEvaluator("testSet", cfg, False, output_dir = cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(cfg, "testSet")
        inference_on_dataset(trainer.model, val_loader, evaluator)
        
        cfg.SOLVER.MAX_ITER += 500
        

if __name__ == '__main__':
    main(sys.argv[1:])
