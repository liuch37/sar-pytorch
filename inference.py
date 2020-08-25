'''
THis is the main inference code.
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set GPU id at the very begining
import argparse
import random
import math
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.multiprocessing import freeze_support
import pdb
# internal package
from dataset import dataset
from dataset.dataset import dictionary_generator
from models.sar import sar
from utils.dataproc import end_cut

# main function:
if __name__ == '__main__':
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument(
        '--worker', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--input', type=str, default='', help='input folder')
    parser.add_argument('--output', type=str, default='predict.txt', help='output file name')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--gpu', type=bool, default=False, help="GPU being used or not")
    
    opt = parser.parse_args()
    print(opt)

    # turn on GPU for models:
    if opt.gpu == False:
        device = torch.device("cpu")
        print("CPU being used!")
    else:
        if torch.cuda.is_available() == True and opt.gpu == True:
            device = torch.device("cuda")
            print("GPU being used!".format(gpu_string))
        else:
            device = torch.device("cpu")
            print("CPU being used!")
    
    # set training parameters
    Height = 48
    Width = 64
    feature_height = Height // 4
    feature_width = Width // 8
    Channel = 3
    voc, char2id, id2char = dictionary_generator()
    output_classes = len(voc)
    embedding_dim = 512
    hidden_units = 512
    layers = 2
    keep_prob = 1.0
    seq_len = 40
    batch_size = opt.batch
    output_path = opt.output
    trained_model_path = opt.model
    input_path = opt.input
    worker = opt.worker

    # load test data
    test_dataset = dataset.test_dataset_builder(Height, Width, input_path)

    # make dataloader
    test_dataloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=int(worker))

    # load model
    print("Create model......")
    model = sar(Channel, feature_height, feature_width, embedding_dim, output_classes, hidden_units, layers, keep_prob, seq_len, device)

    if opt.gpu != -1 and torch.cuda.is_available() == True:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model.to(device)
    
    if trained_model_path != '':
        model.load_state_dict(torch.load(trained_model_path))
    else:
        print("Error: Empty --model path!")
        exit(1)

    if input_path == '':
        print("Error: Empty --input!")
        exit(1)

    if os.path.isfile(output_path):
        os.remove(output_path)

    # run inference
    print("Inference starts......")
    for i, data in enumerate(test_dataloader):
        print("processing for batch index:", i)
        x = data[0] # [batch_size, Channel, Height, Width]
        image_name = data[1] # [batch_size, image_name]
        x = x.to(device)
        model = model.eval()
        predict, _, _, _ = model(x, 0)
        batch_size_current = predict.shape[0]
        pred_choice = predict.max(2)[1] # [batch_size, seq_len]
        with open(output_path, "a") as f:
            for idx in range(batch_size_current):
                # prediction evaluation
                predict_word = end_cut(pred_choice[idx], char2id, id2char)
                # write to output path
                f.write("{} {}\n".format(image_name[idx], predict_word))
    print("Inference done!")