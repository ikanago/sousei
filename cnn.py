# coding: UTF-8

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from unit_layers import Conv, Pool, FC


# 画像認識用ニューラルネットワークのサンプルコード
class myCNN(nn.Module):

    L1_CHANNELS = 4
    L2_CHANNELS = 8
    L3_CHANNELS = 16
    L4_CHANNELS = 16
    L5_CHANNELS = 32
    L6_CHANNELS = 32
    L6_UNITS = 32

    # コンストラクタ
    #   - img_width: 入力画像の横幅
    #   - img_height: 入力画像の縦幅
    #   - img_channels: 入力画像のチャンネル数（グレースケール画像なら1，カラー画像なら3）
    #   - out_units: 認識対象のクラスの数
    def __init__(self, img_width, img_height, img_channels, out_units):
        super(myCNN, self).__init__()
        # 層の定義： ここでは 畳込み層 2 つ，プーリング層 2 つ，全結合層 2 つ とする
        self.conv1 = Conv(in_channels=img_channels, out_channels=self.L1_CHANNELS, kernel_size=30, activation=F.relu)
        self.conv2 = Conv(in_channels=self.L1_CHANNELS, out_channels=self.L2_CHANNELS, kernel_size=30, activation=F.relu)
        self.conv3 = Conv(in_channels=self.L2_CHANNELS, out_channels=self.L3_CHANNELS, kernel_size=30, activation=F.relu)
        self.conv4 = Conv(in_channels=self.L3_CHANNELS, out_channels=self.L4_CHANNELS, kernel_size=30, activation=F.relu)
        self.conv5 = Conv(in_channels=self.L4_CHANNELS, out_channels=self.L5_CHANNELS, kernel_size=30, activation=F.relu)
        self.conv6 = Conv(in_channels=self.L5_CHANNELS, out_channels=self.L6_CHANNELS, kernel_size=30, activation=F.relu)
        self.pool = Pool(method='max')
        self.w = img_width //  16 # プーリング層を 2 回経由するので，特徴マップの横幅は 1/2^2 = 1/4 となる
        self.h = img_height // 16 # 縦幅についても同様
        self.nz = self.w * self.h * self.L4_CHANNELS # 全結合層の直前におけるユニットの総数
        self.fc3 = FC(in_units=self.nz, out_units=self.L6_UNITS, activation=F.relu)
        self.fc4 = FC(in_units=self.L6_UNITS, out_units=out_units, activation=None) # 最終層では一般に活性化関数は入れない

    # 順伝播（最後に softmax なし）
    #   - x: 入力特徴量（ミニバッチ）
    def forward(self, x):
        # コンストラクタで定義した層を順に適用していく
        h = self.conv1(x)
        h = self.pool(h)
        h = self.conv2(h)
        h = self.pool(h)
        h = self.conv3(h)
        h = self.pool(h)
        h = self.conv4(h)
        h = self.pool(h)
        # h = self.conv5(h)
        # h = self.pool(h)
        h = h.view(h.size()[0], -1) # 平坦化：特徴マップを一列に並べ直す（FC層の直前では必ず必要）
        h = self.fc3(h)
        y = self.fc4(h)
        return y

    # 順伝播（最後に softmax あり）
    #   - x: 入力特徴量（ミニバッチ）
    def classify(self, x):
        return F.softmax(self.forward(x), dim=1)
