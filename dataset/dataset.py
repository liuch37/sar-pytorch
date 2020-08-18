'''
This code is to build various dataset for SAR
'''
import string
import cv2
import math
import torch.utils.data as data
import os
import torch
import numpy as np
import sys
import xml.etree.ElementTree as ET
import pdb
import shutil
import json

def dictionary_generator(END='END', PADDING='PAD', UNKNOWN='UNK'):
    '''
    END: end of sentence token
    PADDING: padding token
    UNKNOWN: unknown character token
    '''
    voc = list(string.printable[:-6]) # characters including 9 digits + 26 lower cases + 26 upper cases + 33 punctuations
    
    # update the voc with specifical chars
    voc.append(END)
    voc.append(PADDING)
    voc.append(UNKNOWN)

    char2id = dict(zip(voc, range(len(voc))))
    id2char = dict(zip(range(len(voc)), voc))

    return voc, char2id, id2char

def svt_xml_extractor(img_path, label_path):
    '''
    This code is to extract xml labels from SVT dataset
    Input:
    img_path: path for all image folder
    label_path: xml label path file
    Output:
    dict_img: {image_name: [imgs, labels, lexicon]}
    imgs: list of numpy cropped images with bounding box
    labels: list of string labels
    lexicon: lexicon for this image
    '''
    # create element tree object
    tree = ET.parse(label_path)

    # get root element
    root = tree.getroot()

    # create empty list for news items
    dict_img = {}

    # iterate news items
    for item in root.findall('image'):
        name = item.find('imageName').text.split('/')[-1]
        dict_img[name] = []
        imgs = []
        labels = []
        lexicon = []
        lexicon = item.find('lex').text.split(',')
        rec = item.find('taggedRectangles')
        for r in rec.findall('taggedRectangle'):
            x = int(r.get('x'))
            y = int(r.get('y'))
            w = int(r.get('width'))
            h = int(r.get('height'))
            imgs.append((x,y,w,h))
            labels.append(r.find('tag').text)
        dict_img[name].append(imgs)
        dict_img[name].append(labels)
        dict_img[name].append(lexicon)

    return dict_img

def svt_xml_parser(height, width, total_img_path, xml_path, image_path, label_path):
    # parse xml file and create fully ready dataset
    dictionary = svt_xml_extractor(total_img_path, xml_path)
    if os.path.isdir(image_path):
        shutil.rmtree(image_path)
    os.mkdir(image_path)

    total_img_name = os.listdir(total_img_path)
    # crop and resize
    label_dictionary = {}
    for img_name, items in dictionary.items():
        if img_name in total_img_name:
            IMG_origin = cv2.imread(os.path.join(total_img_path,img_name))
            counter = 0
            for (x,y,w,h), label in zip(items[0], items[1]):
                IMG_crop = IMG_origin[y:y+h,x:x+w,:]
                IMG_crop = cv2.resize(IMG_crop, (width, height))
                crop_name = os.path.join(img_name[:-4]+'_'+str(counter)+'.jpg')
                cv2.imwrite(os.path.join(image_path,crop_name),IMG_crop)
                label_dictionary[crop_name] = label
                counter += 1
        
    # write labels to json file:
    with open(label_path, 'w') as f:
        json.dump(label_dictionary, f)

class svt_dataset_builder(data.Dataset):
    def __init__(self, height, width, total_img_path, xml_path, image_path, label_path):
        pass

    def __getitem__(self, index):
        pass
    def __len__(self):
        pass

# unit test
if __name__ == '__main__':
    
    img_path = '../svt/img/'
    train_xml_path = '../svt/train.xml'
    test_xml_path = '../svt/test.xml'
    train_img_path = '../svt/train_img/'
    train_label_path = '../svt/train_label.json'
    test_img_path = '../svt/test_img/'
    test_label_path = '../svt/test_label.json'
    height = 48 # input height pixel
    width = 64 # input width pixel

    train_dict = svt_xml_extractor(img_path, train_xml_path)
    print("Dictionary for training set is:", train_dict)

    train_dataset = svt_xml_parser(height, width, img_path, train_xml_path, train_img_path, train_label_path)

    test_dataset = svt_xml_parser(height, width, img_path, test_xml_path, test_img_path, test_label_path)