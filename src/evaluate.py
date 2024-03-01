import os
import sys
import time
import random
import config
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import seaborn as sns
import numpy as np
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from utils  import *
from network import *
from Dataset import *
from tqdm import tqdm
from sklearn.metrics import *
from datetime import datetime
from PIL import Image
from torchvision.utils import make_grid, save_image
import matplotlib
import japanize_matplotlib
import csv
import copy


c = {
    'model_name': 'MAE_ViT','seed': 0, 'bs': 32
}

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

def write_LogHeader(log_path):
    #CSVファイルのヘッダー記述
    with open(log_path + "/evaluate_log.csv",'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Modelno','pr-auc'])

def write_Scores(log_path,result_list):
    #リストを1行書き加える。
    with open(log_path + "/evaluate_log.csv",'a') as f:
        writer = csv.writer(f)
        writer.writerow(result_list)

class Evaluater():
    def __init__(self,c):
        self.dataloaders = {}
        self.c = c
        self.now = '{:%y%m%d-%H:%M}'.format(datetime.now())
        self.log_path = os.path.join(config.LOG_DIR_PATH,
                                'evaluate')
        

        #コマンドラインの変数を処理する部分。
        args = len(sys.argv)
        with open(os.path.join(config.LOG_DIR_PATH,'experiment.csv')) as f:
            lines = [s.strip() for s in f.readlines()]
        # 引数がない場合は一番最近のモデルを使用
        if args < 2 :
            target_data = lines[-1].split(',')
        elif isint(sys.argv[1]):
            if int(sys.argv[1])<1:
                print('Use the first data')
                target_data = lines[-1].split(',')
            else:
                try:
                    target_data = lines[int(sys.argv[1])].split(',')
                    self.c['n_per_unit'] = 1 if sys.argv[2] == 'horizontal' else 16
                    self.c['type'] = sys.argv[3].split('=')[1]
                except IndexError:
                    print('It does not exit. Use the first data')
                    target_data = lines[-1].split(',')
                    self.c['n_per_unit'] = 1 if sys.argv[2] == 'horizontal' else 16
        else:
            self.c['n_per_unit'] = 1 if sys.argv[1] == 'horizontal' else 16
            target_data = lines[-1].split(',')

        #上の情報からモデル名を作る。
        self.n_ex = '{:0=2}'.format(int(target_data[1]))
        self.c['model_name'] = target_data[2]
        self.c['n_epoch'] = '{:0=3}'.format(int(target_data[3]))
        temp = self.n_ex+'_'+self.c['model_name']+'_'+self.c['n_epoch']+'ep.pth'
        model_path = os.path.join(config.MODEL_DIR_PATH,temp)

        #モデルの作成。
        self.net = make_model(self.c['model_name'],self.c['n_per_unit'])
        #model_name = 'vit_base_patch16_224'
        #self.net = timm.create_model(model_name,num_classes=config.n_class).to(device)
        self.net.load_state_dict(torch.load(model_path,map_location=device))
        self.net.to(device)
        self.criterion = nn.BCELoss()

    def run(self):
        #各々の部位を整数に変換
        self.c['type'] = CataractTypeToInt(self.c['type'])

        #テストデータセットの用意。
        #pickleファイル、水平断面、回転断面16枚16ラベルを対応づける場合 ()
        pickle_file = config.normal_pkl        
        
        #訓練、検証に分けてデータ分割
        if os.path.exists(pickle_file):
            with open(pickle_file,mode="rb") as f:
                self.dataset = pickle.load(f)
        else :
            self.dataset = load_dataset(self.c['n_per_unit'],self.c['type'],'Add')
            with open(pickle_file,mode="wb") as f:
                pickle.dump(self.dataset,f)

        #self.dataset = load_dataset(self.c['n_per_unit'],self.c['type'],'Add_BlackRects')
        test_id_index,_ = calc_kfold_criterion('test')
        test_index,_ = calc_dataset_index(test_id_index,[],'test',self.c['n_per_unit'])
        test_dataset = Subset(self.dataset['test'],test_index)
        self.dataloaders['test'] = DataLoader(test_dataset,self.c['bs'],
                    shuffle=False,num_workers=os.cpu_count())
        preds,labels,paths,total_loss,accuracy= [],[],[],0,0
        right,notright = 0,0

        self.net.eval()

        for inputs_, labels_,paths_ in tqdm(self.dataloaders['test']):
            inputs_ = inputs_.to(device)
            labels_ = labels_.to(device)

            labels_ = torch.max(labels_,1)[1]


            torch.set_grad_enabled(True)
            outputs_ = self.net(inputs_)
            outputs_ = nn.Softmax(dim=1)(outputs_)

            #loss = self.criterion(outputs_, labels_)

            preds += [outputs_.detach().cpu().numpy()]
            labels += [labels_.detach().cpu().numpy()]
            paths += paths_

            #total_loss += float(loss.detach().cpu().numpy()) * len(inputs_)

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)


        # 作成画像の保存パス,PR-AUCの計算,PR曲線の描画
        model_info = self.n_ex+'_'+self.c['model_name']+'_'+self.c['n_epoch']+'ep'
        fig_path = model_info + '_ep_PRC.png'
        save_fig_path = os.path.join(config.LOG_DIR_PATH,'images',fig_path)

        fig_path = model_info + '_ep_PRBar.png'
        save_fig_path2 = os.path.join(config.LOG_DIR_PATH,'images',fig_path)

        fig_path = model_info + '_ep_F1Bar.png'
        save_fig_path3 = os.path.join(config.LOG_DIR_PATH,'images',fig_path)

        #多クラスの場合、PR-AUCのマクロ平均を計算する
        pr_auc = macro_pr_auc(labels, preds,config.n_class)
        make_PRC(labels,preds,save_fig_path,config.n_class) #クラスiの場合のグラフも作る (縦に3つでとりあえず作ってみる)
        #make_PRBar(labels,preds,save_fig_path2,config.n_class )

        #roc_auc = roc_auc_score(labels, preds,multi_class="ovr",average="macro")
        
        #2クラス分類の場合
        roc_auc = roc_auc_score(labels, preds[:,1])

        #fig_path = model_info + '_ep_ROC.png'
        #save_fig_path = os.path.join(config.LOG_DIR_PATH,'images',fig_path)
        #make_ROC(labels,preds[:,1],save_fig_path)

        #加重平均・四捨五入で予測
        temp_class = np.arange(config.n_class)
        preds = np.sum(preds*temp_class,axis=1)


        #threshold = 0.5
        #preds[preds<threshold] = 0
        #preds[preds>=threshold] = 1

        # 1.6478594e-08
        # 二値より細かい分割の場合
        preds = np.array([round(x) for x in preds])

        #roc_auc = roc_auc_score(labels, preds)

        #preds = np.argmax(preds,axis=1)
        #labels = np.argmax(labels,axis=1)

        r_cnt,tmp,cnt = 0,0,0
        #total_loss /= len(preds)

        # Auc値の計算,ROC曲線の描画

        #for threshold in thresholds:
        #    preds_cpy = copy.deepcopy(preds)
        #    preds_cpy[preds_cpy>threshold] = 1
        #    preds_cpy[preds_cpy<=threshold] = 0
        #    f1 = f1_score(preds_cpy,labels)
        #    f1_list.append(f1)
        #print(f1_list)
        


        #混同行列を作り、ヒートマップで可視化。
        #fig_path = model_info + '_ep_CM.png'
        #save_fig_path = os.path.join(config.LOG_DIR_PATH,'images',fig_path)

        #cm = confusion_matrix(labels,preds)
        #make_ConfusionMatrix(cm,save_fig_path)

        right += (preds == labels).sum()
        notright += len(preds) - (preds == labels).sum()
        accuracy = right / len(test_dataset)
        recall = recall_score(labels,preds,average='macro')
        precision = precision_score(labels,preds,average='macro')
        f1 = f1_score(labels,preds,average='macro')
        #f1 = macro_f1(labels,preds,config.n_class)
        #make_F1Bar(labels,preds,save_fig_path3,config.n_class)

        kappa = cohen_kappa_score(labels,preds,weights='quadratic')

        print('accuracy (macro) :',accuracy)
        print('PR-AUC (macro) :',pr_auc)
        print('ROC-AUC :',roc_auc)
        print('F1 Score (macro)', f1)
        print('Precision (macro) ',precision)
        print('Recall (macro) ',recall)
        print('Kappa',kappa)


        #評価値の棒グラフを作って保存。
        fig,ax = plt.subplots()
        ax.bar(['Acc','AUC-ROC','Recall','Precision','AUC-PRC'],[accuracy,pr_auc,recall,precision,pr_auc],width=0.4,tick_label=['Accuracy','Auc','Recall','Precision','AUC-PRC'],align='center')
        ax.grid(True)
        plt.yticks(np.linspace(0,1,21))
        fig_path = model_info+'ep_graph.png'
        fig.savefig(os.path.join(config.LOG_DIR_PATH,'images',fig_path))


        # 結果をcsvに記録
        result_list = [self.n_ex,pr_auc]
        write_Scores(self.log_path,result_list)


if __name__ == '__main__':
    evaluater = Evaluater(c)
    evaluater.run()