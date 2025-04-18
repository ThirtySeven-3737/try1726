import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import logging
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse
from data.dataloader import read_data
from tqdm import tqdm
import Models
import Configs
import importlib
import random
from utils import metric
import torch.optim.lr_scheduler as lr_scheduler

# 获取当前运行脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 将当前工作目录设置为脚本所在的目录
os.chdir(script_dir)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--modelName', type=str, default='multi_transformer',
                        help='support resnet18/inceptiontime/mobilenetv3_small/mobilenetv3_large/acnet/xresnet1d50/xresnet1d101/ati_cnn')
    parser.add_argument('--datasetName', type=str, default='swell',
                        help='support dreamer/amigos')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers of loading data')
    parser.add_argument('--model_save_dir', type=str, default='results/checkpoint',
                        help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default='results/results/',
                        help='path to save results.')
    parser.add_argument('--data_path', type=str, default='/data2/xu/rebuild/data/new_amigos/',
                        help='path to load data.')
    # parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/',
    #                     help='path to load data.')
    parser.add_argument('--seed', type=int, default="42",
                    help='random seed ')   #use GPU3
    parser.add_argument('--device', type=str, default="cuda:1",
                        help='indicates which gpu will be used.')   
    return parser.parse_args()


def train(model, train_set, val_set, train_type, args):
    device = args.device
    model = model.to(device)
    learning_rate = args.learning_rate
    epoches = args.epoches
    best_model_save_path = args.model_save_dir + "/" + args.modelName + "/" + train_type 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate, weight_decay=0.01)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    best_loss = float('inf')
    best_F1 = float('-inf')
    patience = 0

    for epoch in tqdm(range(epoches), colour='red', desc="Overall_epochs"):
        model.train()
        train_losses = 0
        train_loss = 0
        for x, y in tqdm(train_set, desc="Training"):
            x, y = x.to(device), y.to(device)
            y = y - 1
            y_hat = model(x)
            optimizer.zero_grad()
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses = train_loss / len(train_set)

        print(f'Epoch [{epoch}], Loss: {train_losses:.4f}')

        val_loss, val_F1 = eval(model, val_set, args, train_type, test_mode = "val")
        
        # Updating the learning rate
        scheduler.step(val_F1)

        patience += 1

        # save best checkpoint

        if val_F1 > best_F1:
            best_F1 = val_F1
            if not os.path.exists(best_model_save_path):
                os.makedirs(best_model_save_path)
            torch.save(model.state_dict(), best_model_save_path + "/best_checkpoint.pth")
            print(f'Best model saved at {best_model_save_path}')
            patience = 0   # if we get a new best checkpoint, reset the patience

        if patience >= args.early_stop: 
            print("Early stop mechanism triggers, exit training!")
            break 


def eval(model, eval_data, args, train_type, test_mode):
    device = args.device
    metrics = metric(args)
    model.eval()
    eval_losses = 0
    eval_loss = 0
    all_pre = []
    all_true = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in tqdm(eval_data, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            y = y - 1
            y_hat = model(x)
            loss = criterion(y_hat, y)
            eval_loss += loss.item()
            all_pre.append(y_hat.argmax(1).cpu().numpy())
            all_true.append(y.cpu().numpy())
        
        eval_losses = eval_loss / len(eval_data)
        print(f'{test_mode} Loss: {eval_losses:.4f}')


        all_pre = np.concatenate(all_pre)
        all_true = np.concatenate(all_true)

        metrics_df = metrics.get_all_metric(all_true, all_pre)

        # 打印 DataFrame
        print(metrics_df)

        if test_mode == "test":
            task_res_save_dir = os.path.join(args.res_save_dir, args.datasetName)

            if not os.path.exists(task_res_save_dir):
                os.makedirs(task_res_save_dir)

            # Define the file path
            file_path = os.path.join(task_res_save_dir, f'evaluation_metrics_{train_type}.csv')

            # Check if the file exists
            if os.path.isfile(file_path):
                # Read the existing file
                existing_df = pd.read_csv(file_path)
                # Append the new data to the existing DataFrame
                combined_df = pd.concat([existing_df, metrics_df], ignore_index=True)
            else:
                # If the file does not exist, use the new data as the initial DataFrame
                combined_df = metrics_df

            # Write the combined DataFrame to the file
            combined_df.to_csv(file_path, index=False)
            print("The result has been save!")
        
        else:
            return eval_loss, metrics_df.iloc[0, 4]   

def set_log(args):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    log_file_path = f'logs/{args.modelName}-{args.datasetName}.log'
    # set logging
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)

    for ph in logger.handlers:
        logger.removeHandler(ph)
    # add FileHandler to log file
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    # add StreamHandler to terminal outputs
    formatter_stream = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)
    return logger

#     徐天泽牛逼！
def count_parameters(model):
    answer = 0
    for p in model.parameters():
        if p.requires_grad:
            answer += p.numel()
            # print(p)
    return answer / 1e6

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # all_dataset = ["amigos", "dreamer"]
    all_dataset = ["swell"]
    seeds = [1111, 2222, 3333, 4444, 5555]
    for datasetName in all_dataset:
        for seed in seeds:
            args = parse_args()
            args.seed = seed
            args.datasetName = datasetName
            logger = set_log(args)
            logger.info('Start running seed %s...' %(args.seed))
            logger.info(args)
            data_path = os.path.join(args.data_path, args.datasetName + ".json")
            # data_path = "/data2/xu/rebuild/data/new_amigos/amigos_padding_subject.json"
            setup_seed(args.seed)

            # save current work_dir
            original_cwd = os.getcwd()

            try:
                # dynamic import module
                module_path = f"Models.{args.modelName}"
                config_path = f"Configs.{args.modelName}"

                model_module = importlib.import_module(module_path)
                config_module = importlib.import_module(config_path)

            finally:
                # return current work_dit
                os.chdir(original_cwd)

            config = config_module.Configs(args)

            with open(data_path, 'r') as f:
                data_content = json.load(f)

                train_type = "VA_NN"
                ecg_data, label = read_data().get_data(data_content, train_type)
                train_iter, val_iter, test_iter = read_data().ECG_dataLoader(ecg_data, label, config.args.batch_size)

                # Begin training
                model = model_module.My_model(config.args)
                logger.info(f'The model has {count_parameters(model):.3f}M trainable parameters')

                train(model, train_iter, val_iter, train_type, config.args)

                # begin test in test set
                # load the best checkpoint
                
                model.load_state_dict(torch.load(args.model_save_dir + "/" + args.modelName + "/" + train_type  + "/best_checkpoint.pth"))
                eval(model, test_iter, config.args, train_type, test_mode = "test")



