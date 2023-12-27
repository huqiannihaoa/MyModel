# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
from os import getcwd

sets = ['test']
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
abs_path = os.getcwd()
print(abs_path)


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(image_id):
    in_file = open('./datasets/val/VOCdevkit/VOC2007/Annotations/%s.xml' % (image_id), encoding='UTF-8')
    out_file = open('./datasets/val/VOCdevkit/VOC2007/labels/%s.txt' % (image_id), 'w')
    # in_file = open('./datasets/train/VOCdevkit/VOC2007/Annotations/%s.xml' % (image_id), encoding='UTF-8')
    # out_file = open('./datasets/train/VOCdevkit/VOC2007/labels/%s.txt' % (image_id), 'w')

    # in_file = open('./datasets/val/VOCdevkit/VOC2007/Annotations/%s.xml' % (image_id), encoding='UTF-8')
    # out_file = open('./datasets/val/VOCdevkit/VOC2007/labels/%s.txt' % (image_id), 'w')

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()
for image_set in sets:
    if not os.path.exists('./datasets/val/VOCdevkit/VOC2007/labels/'):
        os.makedirs('./datasets/val/VOCdevkit/VOC2007/labels/')
    image_ids = open(
        './datasets/val/VOCdevkit/VOC2007/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
    list_file = open('./%s.txt' % (image_set), 'w')  # 此处的意思是在VOC2007的子目录中生成图片路径的txt文件
    # print(image_ids)
    for image_id in image_ids:
        list_file.write('./datasets/val/VOCdevkit/VOC2007/JPEGImages/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()

    # if not os.path.exists('./datasets/train/VOCdevkit/VOC2007/labels/'):
    #     os.makedirs('./datasets/train/VOCdevkit/VOC2007/labels/')
    # image_ids = open(
    #     './datasets/train/VOCdevkit/VOC2007/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
    # list_file = open('./%s.txt' % (image_set), 'w')  # 此处的意思是在VOC2007的子目录中生成图片路径的txt文件
    # for image_id in image_ids:
    #     list_file.write('./datasets/train/VOCdevkit/VOC2007/JPEGImages/%s.jpg\n' % (image_id))
    #     convert_annotation(image_id)
    # list_file.close()

    # if not os.path.exists('./datasets/val/VOCdevkit/VOC2007/labels/'):
    #     os.makedirs('./datasets/val/VOCdevkit/VOC2007/labels/')
    # image_ids = open(
    #     './datasets/val/VOCdevkit/VOC2007/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
    # list_file = open('./%s.txt' % (image_set), 'w')  # 此处的意思是在VOC2007的子目录中生成图片路径的txt文件
    # for image_id in image_ids:
    #     list_file.write('./datasets/val/VOCdevkit/VOC2007/JPEGImages/%s.jpg\n' % (image_id))
    #     convert_annotation(image_id)
    # list_file.close()
