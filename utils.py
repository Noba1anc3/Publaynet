import os
import cv2
import json
import random
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

def json_resolve(trainjsonPath, testjsonPath):
    with open(trainjsonPath, 'r') as f:
        train_anno = json.load(f)
        print('Training Set Json File Loaded')
    with open(testjsonPath, 'r') as f:
        test_anno = json.load(f)
        print('Testing Set Json File Loaded')

    train_images = {}
    for image in train_anno['images']:
        train_images[image['id']] = {'file_name': image['file_name'], 'annotations': []}
    for ann in train_anno['annotations']:
        train_images[ann['image_id']]['annotations'].append(ann)

    test_images = {}
    for image in test_anno['images']:
        test_images[image['id']] = {'file_name': image['file_name'], 'annotations': []}
    for ann in test_anno['annotations']:
        test_images[ann['image_id']]['annotations'].append(ann)

    categories = []
    for img in train_anno['categories']:
        categories.append(img['name'])
    print('categories: ', categories)

    return train_images, test_images, categories

def get_textImg_dicts(images, img_dir):

    dataset_dicts = []
    cnt = 0
    for i, (_, image) in enumerate(images.items()):

        record = {}
        filename = os.path.join(img_dir, image["file_name"])
        if not os.path.exists(filename):
            cnt += 1
            continue
        record["file_name"] = filename
        print(filename)

        height, width = cv2.imread(filename).shape[:2]
        record["height"] = height
        record["width"] = width

        annos = image['annotations']
        objs = []

        for anno in annos:
            x = anno["bbox"][0]
            y = anno["bbox"][1]
            w = anno["bbox"][2]
            h = anno["bbox"][3]

            obj = {
                "bbox": [x, y, w, h],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": anno['category_id'] - 1,
                "iscrowd": 0
            }

            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    print('Folder Image Num :', len(dataset_dicts))
    print('Json Image Num :', len(images))
    print('Missed Image Num :', cnt)

    return dataset_dicts


def draw_textImg_dicts(dataset_dicts, smp_num, textImg_metadata):

    for d in random.sample(dataset_dicts, smp_num):
        print('visualize...')
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=textImg_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        img = vis.get_image()[:, :, ::-1]
        path = './output/images/' + d["file_name"].split('/')[3]
        cv2.imwrite(path, img)

    print('visualize finished')

def draw_predImg_dicts(dataset_dicts, smp_num, textImg_metadata, predictor):

    for d in random.sample(dataset_dicts, smp_num):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        print(outputs)
        visualizer = Visualizer(img[:, :, ::-1], metadata=textImg_metadata, scale=0.5)
        vis = visualizer.draw_instance_predictions(outputs['instances'].to('cpu'))
        img = vis.get_image()[:, :, ::-1]
        path = './output/images/' + d["file_name"].split('/')[3]
        cv2.imwrite(path, img)
