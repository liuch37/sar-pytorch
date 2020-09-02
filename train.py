'''
THis is the main training code.
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
from utils.dataproc import performance_evaluate

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
    parser.add_argument('--dataset_type', type=str, default='svt', help="dataset type - svt|iiit5k|syn90k|synthtext")
    parser.add_argument('--gpu', type=bool, default=False, help="GPU being used or not")
    parser.add_argument('--metric', type=str, default='accuracy', help="evaluation metric - accuracy|editdistance")
    
    opt = parser.parse_args()
    print(opt)
    
    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed:", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    
    # turn on GPU for models:
    if opt.gpu == False:
        device = torch.device("cpu")
        print("CPU being used!")
    else:
        if torch.cuda.is_available() == True and opt.gpu == True:
            device = torch.device("cuda")
            print("GPU being used!")
        else:
            device = torch.device("cpu")
            print("CPU being used!")
    
    # set training parameters
    batch_size = opt.batch
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
    epochs = opt.epoch
    worker = opt.worker
    dataset_path = opt.dataset
    dataset_type = opt.dataset_type
    output_path = opt.output
    trained_model_path = opt.model
    eval_metric = opt.metric
    
    # create dataset
    print("Create dataset......")
    if dataset_type == 'svt': # street view text dataset
        img_path = os.path.join(dataset_path, 'img')
        train_xml_path = os.path.join(dataset_path, 'train.xml')
        test_xml_path = os.path.join(dataset_path, 'test.xml')
        train_dataset = dataset.svt_dataset_builder(Height, Width, seq_len, img_path, train_xml_path)
        test_dataset = dataset.svt_dataset_builder(Height, Width, seq_len, img_path, test_xml_path)
    elif dataset_type == 'iiit5k': # IIIT5k dataset
        train_img_path = os.path.join(dataset_path, 'train')
        test_img_path = os.path.join(dataset_path, 'test')
        train_annotation_path = os.path.join(dataset_path, 'traindata.mat')
        test_annotation_path = os.path.join(dataset_path, 'testdata.mat')
        train_dataset = dataset.iiit5k_dataset_builder(Height, Width, seq_len, train_img_path, train_annotation_path)
        test_dataset = dataset.iiit5k_dataset_builder(Height, Width, seq_len, test_img_path, test_annotation_path)
    elif dataset_type == 'syn90k': # Syn90K dataset
        train_img_path = os.path.join(dataset_path, 'train')
        test_img_path = os.path.join(dataset_path, 'test')
        train_dataset = dataset.syn90k_dataset_builder(Height, Width, seq_len, train_img_path)
        test_dataset = dataset.syn90k_dataset_builder(Height, Width, seq_len, test_img_path)
    elif dataset_type == 'synthtext': # SynthText dataset
        train_img_path = os.path.join(dataset_path, 'train')
        test_img_path = os.path.join(dataset_path, 'test')
        annotation_path = os.path.join(dataset_path, 'gt.mat')
        train_dataset = dataset.synthtext_dataset_builder(Height, Width, seq_len, train_img_path, annotation_path)
        test_dataset = dataset.synthtext_dataset_builder(Height, Width, seq_len, test_img_path, annotation_path)
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
    model = sar(Channel, feature_height, feature_width, embedding_dim, output_classes, hidden_units, layers, keep_prob, seq_len, device)

    if trained_model_path != '':
        if torch.cuda.is_available() == True and opt.gpu == True:
            model.load_state_dict(torch.load(trained_model_path, map_location=lambda storage, loc: storage), strict=False)
            model = torch.nn.DataParallel(model).to(device)
        else:
            model.load_state_dict(torch.load(trained_model_path, map_location=lambda storage, loc: storage), strict=False)
    else:
        if torch.cuda.is_available() == True and opt.gpu == True:
            model = torch.nn.DataParallel(model).to(device)
        else:
            model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lmbda = lambda epoch: 0.9**(epoch // 300) if epoch < 13200 else 10**(-2)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)

    num_batch = math.ceil(len(train_dataset) / batch_size)

    # train, evaluate, and save model
    print("Training starts......")
    if eval_metric == 'accuracy':
        best_acc = float('-inf')
    elif eval_metric == 'editdistance':
        best_acc = float('inf')
    else:
        print("Wrong --metric argument, set it to default")
        eval_metric = 'accuracy'
        best_acc = float('-inf')

    for epoch in range(epochs):
        M_list = []
        for i, data in enumerate(train_dataloader):
            x = data[0] # [batch_size, Channel, Height, Width]
            y = data[1] # [batch_size, seq_len, output_classes]
            x, y = x.to(device), y.to(device)
            #print(x.shape, y.shape)
            optimizer.zero_grad()
            model = model.train()
            predict, _, _, _ = model(x, y)
            target = y.max(2)[1] # [batch_size, seq_len]
            #print("Prediction size is:", predict.shape)
            #print("Attention weight size is:", att_weights.shape)
            predict_reshape = predict.permute(0,2,1) # [batch_size, output_classes, seq_len]
            loss = F.nll_loss(predict_reshape, target)
            loss.backward()
            optimizer.step()
            # prediction evaluation
            pred_choice = predict.max(2)[1] # [batch_size, seq_len]
            metric, metric_list, predict_words, labeled_words = performance_evaluate(pred_choice.detach().cpu().numpy(), target.detach().cpu().numpy(), voc, char2id, id2char, eval_metric)
            M_list += metric_list
            print('[Epoch %d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), metric))
            #print("predict prob:", predict[0][0])
            #print("predict words:", predict_words[0])
            #print("labeled words:", labeled_words[0])
        train_acc = float(sum(M_list)/len(M_list))
        print("Epoch {} average train accuracy: {}".format(epoch, train_acc))

        scheduler.step()

        # Validation
        print("Testing......")
        with torch.set_grad_enabled(False):
            M_list = []
            for i, data in enumerate(test_dataloader):
                x = data[0] # [batch_size, Channel, Height, Width]
                y = data[1] # [batch_size, seq_len, output_classes]
                x, y = x.to(device), y.to(device)
                model = model.eval()
                predict, _, _, _ = model(x, y)
                # prediction evaluation
                pred_choice = predict.max(2)[1] # [batch_size, seq_len]
                target = y.max(2)[1] # [batch_size, seq_len]
                metric, metric_list, predict_words, labeled_words = performance_evaluate(pred_choice.detach().cpu().numpy(), target.detach().cpu().numpy(), voc, char2id, id2char, eval_metric)
                M_list += metric_list
            test_acc = float(sum(M_list)/len(M_list))
            #print("Test predict words:", predict_words[0])
            #print("Test labeled words:", labeled_words[0])
            print("Epoch {} average test accuracy: {}".format(epoch, test_acc))
            with open(os.path.join(output_path,'statistics.txt'), 'a') as f:
                f.write("{} {}\n".format(train_acc, test_acc))
            if eval_metric == 'accuracy':
                if test_acc >= best_acc:
                    print("Save current best model with accuracy:", test_acc)
                    best_acc = test_acc
                    if torch.cuda.is_available() == True and opt.gpu == True:
                        torch.save(model.module.state_dict(), '%s/model_best.pth' % (output_path))
                    else:
                        torch.save(model.state_dict(), '%s/model_best.pth' % (output_path))
            elif eval_metric == 'editdistance':
                if test_acc <= best_acc:
                    print("Save current best model with accuracy:", test_acc)
                    best_acc = test_acc
                    if torch.cuda.is_available() == True and opt.gpu == True:
                        torch.save(model.module.state_dict(), '%s/model_best.pth' % (output_path))
                    else:
                        torch.save(model.state_dict(), '%s/model_best.pth' % (output_path))
    print("Best test accuracy is:", best_acc)