'''
THis is the main training code.
'''
import argparse
import os
import random
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

# main function:
if __name__ == '__main__':
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch', type=int, default=32, help='input batch size')
    parser.add_argument(
        '--worker', type=int, default=4, help='number of data loading workers')
    parser.add_argument(
        '--epoch', type=int, default=250, help='number of epochs')
    parser.add_argument('--output', type=str, default='str', help='output folder name')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--dataset_type', type=str, default='svt', help="dataset type")
    parser.add_argument('--gpu', nargs='+', type=int, default=-1, help="GPU indices")
    
    opt = parser.parse_args()
    print(opt)
    
    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed:", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    
    # turn on GPU for models:
    if opt.gpu == -1:
        device = torch.device("cpu")
        print("CPU being used!")
    else:
        if torch.cuda.is_available() == True:
            gpu_string = ','.join([str(item) for item in opt.gpu])
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_string
            device = torch.device("cuda")
            print("GPU {} being used!".format(gpu_string))
        else:
            device = torch.device("cpu")
            print("CPU being used!")
    
    # set training parameters
    batch_size = opt.batch
    Height = 48
    Width = 64
    Channel = 3
    voc, char2id, id2char = dictionary_generator()
    output_classes = len(voc)
    embedding_dim = 512
    hidden_units = 512
    layers = 2
    keep_prob = 1.0
    seq_len = 40
    epochs = opt.epoch
    worker = opt.worker
    dataset_path = opt.dataset
    dataset_type = opt.dataset_type
    output_path = opt.output
    trained_model_path = opt.model
    
    # create dataset
    print("Create dataset......")
    if dataset_type == 'svt': # street view text dataset:
        img_path = os.path.join(dataset_path,'img')
        train_xml_path = os.path.join(dataset_path,'train.xml')
        test_xml_path = os.path.join(dataset_path,'test.xml')
        train_dataset = dataset.svt_dataset_builder(Height, Width, seq_len, img_path, train_xml_path)
        test_dataset = dataset.svt_dataset_builder(Height, Width, seq_len, img_path, test_xml_path)
    else:
        print("Not supported yet!")
        exit(1)
    
    # make dataloader
    train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=int(worker))
        
    test_dataloader = torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=int(worker))
        
    print("Length of train dataset is:", len(train_dataset))
    print("Length of test dataset is:", len(test_dataset))
    print("Number of output classes is:", train_dataset.output_classes)
    
    # make model output folder
    try:
        os.makedirs(output_path)
    except OSError:
        pass
    
    # create model
    print("Create model......")
    model = sar(Channel, embedding_dim, output_classes, hidden_units, layers, keep_prob, seq_len, training=True)
    
    if opt.gpu != -1 and torch.cuda.is_available() == True:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model.to(device)
    
    if trained_model_path != '':
        model.load_state_dict(torch.load(trained_model_path))
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
    num_batch = len(train_dataset) // batch_size
    
    # train, evaluate, and save model
    for epoch in range(epochs):
        scheduler.step()
        for i, data in enumerate(train_dataloader):
            x = data[0] # [batch_size, Channel, Height, Width]
            y = data[1] # [batch_size, seq_len, output_classes] 
            print(x.shape)
            print(y.shape)
    
            #predict, att_weights = model(x,y)
            #print("Prediction size is:", predict.shape)
            #print("Attention weight size is:", att_weights.shape)