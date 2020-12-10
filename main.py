#! /usr/bin/env python
import os
import argparse
import datetime

import torch
import torchtext.data as data
import torchtext.datasets as datasets
import model
import train
import mydatasets
import pandas as pd
import glob
import time

from GPyOpt.methods import BayesianOptimization


parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=256, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-baye', type=bool, default=False)
parser.add_argument('-test', action='store_true', default=False, help='train or test')
#path
parser.add_argument('-input_path', type=str, default='datas/sum_sampledata_asr_g2pk.csv') # clean_g2pk, clean_g2pk_withEDA, EDA_g2pk_concat
args = parser.parse_args()



domain = [{'name': 'lr_rate',
          'type': 'continuous',
          'domain': (0.001, 0.01),
           'dimensionality': 1},
            {'name': 'dropout',
          'type': 'continuous',
          'domain': (0.4, 1),
           'dimensionality': 1}]
hyper = []
# load SST dataset
def sst(text_field, label_field,  **kargs):
    train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained=True)
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                                        (train_data, dev_data, test_data), 
                                        batch_sizes=(args.batch_size, 
                                                     len(dev_data), 
                                                     len(test_data)),
                                        **kargs)
    return train_iter, dev_iter, test_iter 


# load MR dataset
def mr(text_field, label_field, **kargs):
    train_data, dev_data = mydatasets.MR.splits(text_field, label_field) #이거로 train_data, test_data를 만드는거
    print('여기확인')
    print(vars(train_data[0]))
    print(vars(dev_data[2]))
    print(train_data.fields.items())
    text_field.build_vocab(train_data, dev_data) # 단어 집합을 생성
    label_field.build_vocab(train_data, dev_data)
    '''
    train_iter, dev_iter = data.Iterator.splits(
                                (train_data, dev_data), 
                                batch_sizes=(args.batch_size, len(dev_data)),
                                **kargs)
    '''
    train_iter = data.Iterator(dataset=train_data, batch_size=args.batch_size)
    dev_iter = data.Iterator(dataset=dev_data, batch_size=len(dev_data))

    print('훈련 데이터의 미니 배치 수 : {}'.format(len(train_iter)))
    print('테스트 데이터의 미니 배치 수 : {}'.format(len(dev_iter)))
    return train_iter, dev_iter

def msw_text(text_field, label_field, train_path, **kargs):
    train_data, dev_data = mydatasets.MR_2.splits(text_field, label_field, train_path, shuffle=True)  # 이거로 train_data, test_data를 만드는거

    text_field.build_vocab(train_data, dev_data)  # 단어 집합을 생성
    label_field.build_vocab(train_data, dev_data)

    train_iter, dev_iter = data.Iterator.splits(
                                (train_data, dev_data), 
                                batch_sizes=(args.batch_size, len(dev_data)),
                                **kargs)
    '''
    train_iter = data.Iterator(dataset=train_data, batch_size=args.batch_size)
    dev_iter = data.Iterator(dataset=dev_data, batch_size=len(dev_data))
    '''
    print('check Vocabulary', text_field.vocab.stoi)
    print('훈련 데이터의 미니 배치 수 : {}'.format(len(train_iter)))
    print('테스트 데이터의 미니 배치 수 : {}'.format(len(dev_iter)))
    return train_iter, dev_iter
'''
def baye(arg):
    learning_rate = arg[0, 0]
    drop_out = arg[0, 1]
    args.lr = learning_rate
    args.dropout = drop_out

    cnn = model.CNN_Text(args)
    if args.snapshot is not None:
        print('\nLoading model from {}...'.format(args.snapshot))
        cnn.load_state_dict(torch.load(args.snapshot))

    if args.cuda:
        torch.cuda.set_device(args.device)
        cnn = cnn.cuda()

    global hyper
    baye_value = train.train(train_iter, dev_iter, cnn, args)

    return baye_value
'''

def text_cnn_train(args, train_path):
    # load data
    print("\nLoading data...")
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    #train_iter, dev_iter = mr(text_field, label_field, device=-1, repeat=False)
    train_iter, dev_iter = msw_text(text_field, label_field, train_path, device=-1, repeat=False)

    # batch = next(iter(train_iter))
    # print(type(batch))
    # print(batch.text)

    # train_iter, dev_iter, test_iter = sst(text_field, label_field, device=-1, repeat=False)


    # update args and print
    args.embed_num = len(text_field.vocab) # .vocab을 해주면 단어 집합을 만들어 주는거 같다. 일단 추정
    args.class_num = len(label_field.vocab) - 1
    args.cuda = (not False) and torch.cuda.is_available()
    kerns = '3,4,5'
    args.kernel_sizes = [int(k) for k in kerns.split(',')]
    re_train_path = train_path.split('/')[1][:-4]
    save_path = os.path.join(args.save_dir, re_train_path)


    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))


    # train or predict
    if args.baye:
        print('옵티마이즈 세팅')
        myBopt = BayesianOptimization(f=baye, domain=domain, initial_design_numdata=5)
        print('옵티마이즈 시작')
        myBopt.run_optimization(max_iter=10)
        print('옵티마이즈 결과 : ', myBopt.x_opt)
        print('최적의 하이퍼 파라미터의 결과의 파라미터', args.lr, args.dropout)

    else:
        try:
            cnn = model.CNN_Text(args)
            if args.snapshot is not None:
                print('\nLoading model from {}...'.format(args.snapshot))
                cnn.load_state_dict(torch.load(args.snapshot))

            if args.cuda:
                torch.cuda.set_device(args.device)
                cnn = cnn.cuda()

            train.train(train_iter, dev_iter, cnn, args, save_path)
        except KeyboardInterrupt:
            print('\n' + '-' * 89)
            print('Exiting from training early')

train_path = []
train_data_path = 'traindata_v4'
file_list = glob.glob(os.path.join(train_data_path, '*'))
file_list.sort()
print(file_list)
time.sleep(3)
for file in file_list:
    text_cnn_train(args, file)
    print('finish')