import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import csv
import torch
from cnn import myCNN
from utils import show_image
from data_io import read_image_list, load_single_image, load_images


# データセットの指定
DATA_DIR = './dataset/facesdb_prepared/' # データフォルダのパス
IMAGE_LIST_EV = DATA_DIR + 'test_list.csv' # 評価用データ

# 学習済みモデルが保存されているフォルダのパス
MODEL_DIR = './emotion_models/'

def convert_to_feeling(n):
    assert(0 <= n < 7)
    if n == 0:
        return 'Neutral'
    elif n == 1:
        return 'Joy'
    elif n == 2:
        return 'Negative'
    else:
        return 'Surprise'

# エントリポイント
if __name__ == '__main__':

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description = 'CNN Model for MNIST Image Recognition (Prediction)')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', '-b', default=100, type=int, help='learning minibatch size')
    parser.add_argument('--in_filepath', '-i', default='', type=str, help='input image file path')
    parser.add_argument('--model', '-m', default='', type=str, help='file path of trained model')
    args = parser.parse_args()

    # コマンドライン引数のチェック
    if args.model is None or args.model == '':
        print('error: model file is not specified.', file=sys.stderr)
        exit()

    # デバイスの設定
    dev_str = 'cuda:{0}'.format(args.gpu) if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    dev = torch.device(dev_str)

    # オプション情報の設定・表示
    in_filepath = args.in_filepath # 入力画像のファイルパス
    batchsize = max(1, args.batchsize) # バッチサイズ
    model_filepath = args.model # 学習済みモデルのファイルパス
    print('device: {0}'.format(dev_str), file=sys.stderr)
    if in_filepath == '':
        print('batchsize: {0}'.format(batchsize), file=sys.stderr)
    else:
        print('input file: {0}'.format(in_filepath), file=sys.stderr)
    print('model file: {0}'.format(model_filepath), file=sys.stderr)
    print('', file=sys.stderr)

    # 画像の縦幅・横幅・チャンネル数の設定
    width = 240 # MNIST文字画像の場合，横幅は 28 pixels
    height = 240 # MNIST文字画像の場合，縦幅も 28 pixels
    channels = 3 # MNIST文字画像はグレースケール画像なので，チャンネル数は 1
    color_mode = 0 if channels == 1 else 1

    # ラベル名とラベル番号を対応付ける辞書をロード
    with open(MODEL_DIR + 'labeldict.pickle', 'rb') as f:
        labeldict = pickle.load(f)
    with open(MODEL_DIR + 'labelnames.pickle', 'rb') as f:
        labelnames = pickle.load(f)
    n_classes = len(labelnames)

    # 学習済みの画像認識器をロード
    model = myCNN(width, height, channels, n_classes)
    model.load_state_dict(torch.load(model_filepath))
    model = model.to(dev)
    model.eval()

    # 入力画像に対し認識処理を実行
    if in_filepath == '':

        ### ファイル名を指定せずに実行した場合・・・全評価用データに対する識別精度を表示 ###

        labels_ev, imgfiles_ev, labeldict, dmy = read_image_list(IMAGE_LIST_EV, DATA_DIR, dic=labeldict) # 評価用データの読み込み
        n_samples_ev = len(imgfiles_ev) # 評価用データの総数
        n_failed = 0
        # row: expected, col: actual
        eval_matrix = [[0 for j in range(7)] for i in range(7)]
        for i in range(0, n_samples_ev, batchsize):
            offset = min(n_samples_ev, i + batchsize)
            x = torch.tensor(load_images(imgfiles_ev, ids=np.arange(i, offset), mode=color_mode), device=dev)
            t = labels_ev[i : i + batchsize]
            y = model.classify(x)
            y_cpu = y.to('cpu').detach().numpy().copy()
            recognized = np.argmax(y_cpu, 1)
            for j in range(len(t)):
                eval_matrix[t[j]][recognized[j]] += 1
            n_failed += np.count_nonzero(np.argmax(y_cpu, axis=1) - t)
            del y_cpu
            del y
            del x
            del t
        acc = (n_samples_ev - n_failed) / n_samples_ev
        print('accuracy = {0:.2f}%'.format(100 * acc), file=sys.stderr)
        for i in range(7):
            for j in range(7):
                if i == j:
                    continue
                if eval_matrix[i][j] > 3:
                    print("expected: {} actual: {}".format(i, j))

    else:

        ### 画像ファイル名を指定して実行した場合・・・指定された画像に対する認識結果を表示 ###

        img = np.asarray([load_single_image(in_filepath, mode=color_mode)]) # 入力画像を読み込む
        show_image(img[0], title='input image', mode=color_mode) # 入力画像を表示
        image_label = int(in_filepath[-10:-8])
        print('expected result: {0}'.format(convert_to_feeling(image_label)), file=sys.stderr)
        x = torch.tensor(img, device=dev)
        y = model.classify(x)
        y_cpu = y.to('cpu').detach().numpy().copy()
        recognized_label = int(labelnames[np.argmax(y_cpu)])
        print('recognition result: {0}'.format(convert_to_feeling(recognized_label)), file=sys.stderr)
        del y_cpu
        del y
        del x

    print('', file=sys.stderr)

