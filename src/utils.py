import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import pickle
import config
import statistics
import copy
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import torch.nn as nn
import models_mae
import time
from sklearn.metrics import *

#2値分類
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

#class Focal_MultiLabel_Loss(nn.Module):
#    def __init__(self, gamma,weights):
#      super(Focal_MultiLabel_Loss, self).__init__()
#      self.gamma = gamma
#      self.weights = weights
#
#    def forward(self, outputs, targets):
#        focal_loss = torch.zeros(1).to(device)
#        for i,output in enumerate(outputs):
#            index = torch.argmax(targets,dim=1)
#            self.pt = outputs[i][index[i]].to(device)
#            self.weight = -self.weights[index[i]]
#            focal_loss +=  self.weight * ((1-self.pt)**self.gamma) * torch.log(self.pt)
#
##        print(self.pt,self.weight)#
#
#        return focal_loss

class Focal_MultiLabel_Loss(nn.Module):
    def __init__(self, gamma,weights):
      super(Focal_MultiLabel_Loss, self).__init__()
      self.gamma = gamma
      self.celoss = nn.CrossEntropyLoss(reduction='none')
      self.celoss_withweights = nn.CrossEntropyLoss(weight = weights, reduction='none')

    def forward(self, outputs, targets):
      #outputs = torch.where(outputs>1, torch.ones_like(outputs), outputs)  # 1を超える場合には1にする
      outputs = torch.nan_to_num(outputs, nan=0.5)
      #print(outputs,targets)
      ce = self.celoss(outputs, targets)
      ce_withweights = self.celoss_withweights(outputs,targets)
      ce_exp = torch.exp(-ce)
      focal_loss = ((1-ce_exp)**self.gamma) * ce_withweights
      return focal_loss.mean()



class NormalSampler:
    def __init__(self,dataset,batch_size,shuffle=True):
        l = len(dataset)
        self.imgs = torch.squeeze(torch.stack([dataset[i][0].detach().cpu() for i in range(l)],dim=0),dim=1)
        self.labels = torch.squeeze(torch.stack([dataset.pick_label(i).detach().cpu() for i in range(len(dataset))],dim=0),dim=1)
        self.batch_size = batch_size * 2
        self.indices_count = 0

        self.indices = list(range(len(dataset)))
        np.random.shuffle(self.indices)

    def __iter__(self):
        while self.indices_count + self.batch_size <= len(self.indices):
            indices = self.indices[self.indices_count:self.indices_count + self.batch_size]
            yield self.imgs[indices],self.labels[indices],0
            self.indices_count += self.batch_size

    def __len__(self):
        return len(self.indices // self.batch_size)



#データバッチの中を陽性と陰性1:1にして、オーバーサンプリングして集める
#ここを画像とラベルを別々で渡すのではなく、Datasetで渡す、という仕様に変えれば良い。
#SamplerというかもうDataloader

class TripleOverSampler:
    def __init__(self,dataset,n_samples):
        imgs = torch.empty(0)
        labels = torch.empty(0)

        #datasetをtorch.Tensorとして取り出す
        for i in range(len(dataset)):
            img = dataset[i][0]
            imgs = torch.cat((imgs,img),0)
            label = torch.Tensor([np.argmax(dataset[i][1].detach().cpu().numpy())])
            labels = torch.cat((labels,label))

        self.features = imgs
        self.labels = labels

        label_counts = np.bincount(labels)
        major_label = label_counts.argmax()
        mid_label = np.argsort(label_counts,axis=1)
        minor_label = label_counts.argmin()

        print(label_counts,major_label,mid_label,minor_label)

        print(major_label,mid_label,minor_label)
        self.major_indices = np.where(labels == major_label)[0]
        self.mid_indices = np.where(labels == mid_label)[0]
        self.minor_indices = np.where(labels == minor_label)[0]

        np.random.shuffle(self.major_indices)
        np.random.shuffle(self.minor_indices)

        self.used_indices = 0  #(多数クラスの中で使用したインデックスの数)
        self.count = 0
        self.n_samples = n_samples
        self.batch_size = self.n_samples * 2

    def __iter__(self):
        self.count = 0
        self.used_indices = 0
        while self.count + self.n_samples < len(self.major_indices):
            # 多数派データ(major_indices)からは順番に選び出し
            # 少数派データ(minor_indices)からはランダムに選び出す操作を繰り返す
            indices = self.major_indices[self.used_indices:self.used_indices + self.n_samples].tolist() + np.random.choice(self.mid_indices, self.n_samples, replace=False).tolist() + np.random.choice(self.minor_indices, self.n_samples, replace=False).tolist()
            np.random.shuffle(indices)

            labels = torch.empty(0)
            for index in indices:
                label = torch.eye(config.n_class)[self.labels[index].detach().cpu().numpy()]
                label = label.unsqueeze(0)
                labels = torch.cat((labels,label))

            yield torch.tensor(self.features[indices]), labels,0

            self.used_indices += self.n_samples
            self.count += self.n_samples

    def __len__(self):
        return len(self.major_indices)//self.n_samples + 1

class BinaryOverSampler:
    def __init__(self,dataset,n_samples):

        #for i in range(len(dataset)):
        #    img = dataset[i][0]
        #    imgs = torch.cat((imgs,img),0)
            #label = dataset[i][1]
        #    label = torch.Tensor([np.argmax(dataset[i][1].detach().cpu().numpy())])
        #    labels = torch.cat((labels,label))


        #self.features = imgs
        #self.labels = labels
        #self.dataset = dataset

        l = len(dataset)
        self.imgs = torch.squeeze(torch.stack([dataset[i][0].detach().cpu() for i in range(len(dataset))],dim=0),dim=1)
        self.labels = torch.squeeze(torch.stack([dataset.pick_label(i).detach().cpu() for i in range(len(dataset))],dim=0),dim=1)
        #self.labels = torch.squeeze(torch.stack([dataset[i][1].detach().cpu() for i in range(len(dataset))],dim=0),dim=1)
        labels = []
        for i in range(l):
            if all(torch.argmax(dataset.pick_label(i)) == torch.Tensor([1])):
                labels += [1]
            else :
                labels += [0]

        label_counts = np.bincount(np.array(labels))
        major_label = label_counts.argmax()
        minor_label = label_counts.argmin()
        
        self.major_indices = np.where(labels == major_label)[0]
        self.minor_indices = np.where(labels == minor_label)[0]
        
        np.random.shuffle(self.major_indices)
        np.random.shuffle(self.minor_indices)
        
        self.used_indices = 0
        self.count = 0
        self.n_samples = n_samples
        self.batch_size = self.n_samples * 2

    def __iter__(self):
        self.count = 0
        self.used_indices = 0
        while self.count + self.n_samples < len(self.major_indices):
            # 多数派データ(major_indices)からは順番に選び出し
            # 少数派データ(minor_indices)からはランダムに選び出す操作を繰り返す
            indices = self.major_indices[self.used_indices:self.used_indices + self.n_samples].tolist() + np.random.choice(self.minor_indices, self.n_samples, replace=False).tolist()
            np.random.shuffle(indices)

            #labels = torch.empty(0)
            #for index in indices:
#                np.argmax(self.labels[index].detach().cpu().numpy())
            #    label = torch.eye(config.n_class)[self.labels[index].detach().cpu().numpy()]
            #    label = label.unsqueeze(0)
            #    labels = torch.cat((labels,label))

 #           imgs = torch.squeeze(torch.stack([self.dataset[i][0].detach().cpu() for i in indices],dim=0),dim=1)
 #           labels = torch.squeeze(torch.stack([self.dataset[i][1].detach().cpu() for i in indices],dim=0),dim=1)      

            yield self.imgs[indices],self.labels[indices],0

            #ここじゃないか？
            self.used_indices += self.n_samples
            self.count += self.n_samples

    def __len__(self):
        return len(self.major_indices)//self.n_samples + 1

#データバッチの中を陽性と陰性1:1にして、アンダーサンプリングして集める
class BinaryUnderSampler:
    def __init__(self,dataset,n_samples):

        start_time = time.time()
        '''
        imgs = torch.empty(0)
        labels = torch.empty(0)

        for i in range(len(dataset)):
            img = dataset[i][0]
            imgs = torch.cat((imgs,img),0)
            #label = dataset[i][1]
            label = torch.Tensor([np.argmax(dataset[i][1].detach().cpu().numpy())])
            labels = torch.cat((labels,label))

        self.features = imgs
        self.labels = labels
        '''

        l = len(dataset)
        self.imgs = torch.squeeze(torch.stack([dataset[i][0].detach().cpu() for i in range(l)],dim=0),dim=1)
        self.labels = torch.squeeze(torch.stack([dataset.pick_label(i).detach().cpu() for i in range(len(dataset))],dim=0),dim=1)
        #self.labels = torch.squeeze(torch.stack([dataset[i][1].detach().cpu() for i in range(len(dataset))],dim=0),dim=1)
        labels = []
        for i in range(l):
            if all(torch.argmax(dataset.pick_label(i)) == torch.Tensor([1])):
                labels += [1]
            else :
                labels += [0]
        
        label_counts = np.bincount(labels)
        major_label = label_counts.argmax()
        minor_label = label_counts.argmin()
        
        self.major_indices = np.where(labels == major_label)[0]
        self.minor_indices = np.where(labels == minor_label)[0]
        
        np.random.shuffle(self.major_indices)
        np.random.shuffle(self.minor_indices)
        
        self.used_indices = 0
        self.count = 0
        self.n_samples = n_samples
        self.batch_size = self.n_samples * 2

        end_time = time.time()
        print("所要時間",end_time-start_time)

    def __iter__(self):
        self.count = 0
        self.used_indices = 0
        while self.count + self.n_samples < len(self.minor_indices):
            # 多数派データ(major_indices)、# 少数派データ(minor_indices)ともに順番に選び出す
            indices = self.major_indices[self.used_indices:self.used_indices + self.n_samples].tolist() + self.minor_indices[self.used_indices:self.used_indices + self.n_samples].tolist()
            np.random.shuffle(indices)

            #labels = torch.empty(0)
            #for index in indices:
            #    np.argmax(self.labels[index].detach().cpu().numpy())
            #    label = torch.eye(config.n_class)[self.labels[index].detach().cpu().numpy()]
            #    label = label.unsqueeze(0)
            #    labels = torch.cat((labels,label))

            #imgs = torch.squeeze(torch.stack([self.dataset[i][0].detach().cpu() for i in indices],dim=0),dim=1)
            #labels = torch.squeeze(torch.stack([self.dataset[i][1].detach().cpu() for i in indices],dim=0),dim=1)        

            yield self.imgs[indices],self.labels[indices],0
            
            self.used_indices += self.n_samples
            self.count += self.n_samples * 2

        #長さは少数の陽性サンプルの長さと同じ
        def __len__(self):
            return len(self.minor_indices)//self.n_samples + 1


#class EarlyStopping:
#    """earlystoppingクラス"""
#
#    def __init__(self, patience=5, verbose=False, delta=0,path='./model/checkpoint_model.pth'):
#        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""
#
#        self.patience = patience    
#        self.verbose = verbose      
#        self.counter = 0
#        self.epoch = 0            
#        self.best_score = None     
#        self.early_stop = False 
#        self.val_pr_auc_max = 0  
#        self.path = path
#        self.delta = delta       

 #   def __call__(self, val_pr_auc, model,epoch):
 #       """
 #       特殊(call)メソッド
 #       実際に学習ループ内で最小lossを更新したか否かを計算させる部分
 #       """
 #       score = val_pr_auc
#
 #       if self.best_score is None: 
#            self.best_score = score   
#            self.checkpoint(val_pr_auc,model,epoch)  
#        elif score < self.best_score + self.delta:  #ベストスコアを更新できなかった場合
#            self.counter += 1   #ストップカウンタを+1
#            if self.verbose:  #表示を有効にした場合は経過を表示
#                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
#            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
#                self.early_stop = True
#        else:  #ベストスコアを更新した場合
#            self.best_score = score  
#            self.epoch = epoch
#            self.checkpoint(val_pr_auc, model,epoch)  #モデルを保存してスコア表示
#            self.counter = 0  #ストップカウンタリセット
#
#    def checkpoint(self, val_pr_auc, model,epoch):
#        '''ベストスコア更新時に実行されるチェックポイント関数'''
#        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
#            print(f'Validation mae decreased ({self.val_pr_auc_max:.6f} --> {val_pr_auc:.6f}).  Saving model ...')
#        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        #torch.save(model.module.state_dict(),'./model/evaluate.pth') #評価用のpathにベストモデルを保存
#        self.val_pr_auc_max = val_pr_auc 

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(a):
    exp_a = np.exp(a) # 分子
    sum_exp_a = np.sum(exp_a) # 分母
    y = exp_a / sum_exp_a # 式(3.10)
    return y

def prepare_model(chkpt_dir, arch='mae_vit_base_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location=device)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def iterate(d, param={}):
    d, param = d.copy(), param.copy()
    d_list = []

    for k, v in d.items():
        if isinstance(v, list):
            for vi in v:
                d[k], param[k] = vi, vi
                d_list += iterate(d, param)
            return d_list

        if isinstance(v, dict):
            add_d_list = iterate(v, param)
            if len(add_d_list) > 1:
                for vi, pi in add_d_list:
                    d[k] = vi
                    d_list += iterate(d, pi)
                return d_list

    return [[d, param]]

def isint(s):
    try:
        int(s)
    except ValueError:
        return False
    else:
        return True

def init_weights(m):
    if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data,a=math.sqrt(5))
        if m.bias is not None:
            fan_in,_ = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias.data,-bound,bound)
            
    elif isinstance(m,nn.BatchNorm2d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)

#IDをもとに、その画像を取り出すときのデータセットでのインデックスを返す
def calc_dataset_index(learning_id_index,valid_id_index,mode,n_per_unit):
    data_csv = pd.read_csv(config.train_info_list) if mode=='train' else pd.read_csv(config.test_info_list)
    #12月の分割で試すときはこっちのpath
    path = '../medicaldata/txt/train_ids.pkl' if mode=='train' else '../medicaldata/txt/test_ids.pkl'
    
    #MD3名の分割で試すときはこっちのpath
    #path = '../medicaldata/txt/train_ids2.pkl' if mode=='train' else '../medicaldata/txt/test_ids2.pkl'
    
    with open(path,'rb') as f:
        ids = pickle.load(f)
    #traing id size
    ts = len(ids)
    y = [0]*ts

    for i in range(ts):
        temp = data_csv[data_csv['ID'] == ids[i]]['C']
        y[i] = int(temp.values[0])

    indexs = list(range(ts))
    learning_index,valid_index,tail = [],[],0

    #回転画像の場合は、n_per_unit = 16,水平画像の場合は、n_per_unit = 1
    n_per_unit = 1

    for i in range(ts):
        temp_id = ids[i]
        size = len(data_csv[data_csv['ID'] == temp_id])
        if i in learning_id_index:
            learning_index += list(range(tail,tail+n_per_unit*size))
        else:
            valid_index += list(range(tail,tail+n_per_unit*size))
        tail += n_per_unit*size

    return learning_index,valid_index

def calc_kfold_criterion(mode):
    data_csv = pd.read_csv(config.train_info_list) if mode=='train' else pd.read_csv(config.test_info_list)
    #12月の分割で試すときはこっちのpath
    path = '../medicaldata/txt/train_ids.pkl' if mode=='train' else '../medicaldata/txt/test_ids.pkl'
    
    #MD3名の分割で試すときはこっちのpath
    #path = '../medicaldata/txt/train_ids2.pkl' if mode=='train' else '../medicaldata/txt/test_ids2.pkl'
    
    with open(path,'rb') as f:
        ids = pickle.load(f)
    #traing id size
    ts = len(ids)
    y = [0]*ts            

    for i in range(ts):
        temp = data_csv[data_csv['ID'] == ids[i]]['C']
        y[i] = int(temp.values[0])

    indexs = list(range(ts))

    return indexs,y


def CataractTypeToInt(cataract_type):
    if cataract_type == 'N':
        return 5
    elif cataract_type == 'C':
        return 7
    elif cataract_type == 'P':
        return 8

def make_ROC(labels,preds,save_fig_path):
        fpr,tpr,threshold = roc_curve(labels,preds)
        fig,ax = plt.subplots(figsize=(6,6))
        plt.plot(fpr,tpr)
        plt.legend()
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        fig.savefig(save_fig_path)

def make_ConfusionMatrix(cm,save_fig_path):
        fig,ax = plt.subplots(figsize=(6,6))
        sns.set(font="IPAexGothic",font_scale=1.8)  
        sns.heatmap(cm,square=True,cbar=True,annot=True,cmap='Blues',fmt='d')
        ax.set_ylabel('正解値',fontsize=18)
        #ax.set_xticklabels(['1','2,3','4≦'],fontsize=20)
        #ax.set_yticklabels(['1','2,3','4≦'],fontsize=20)
        ax.set_xticklabels(['N1','N2,N3','≧N4'],fontsize=20)
        ax.set_yticklabels(['N1','N2,N3','≧N4'],fontsize=20)
        ax.set_xlabel('予測値',fontsize=18)
        ax.set_title('Confusion Matrix',fontsize=20)
        fig.savefig(save_fig_path)

def make_PRC(labels,preds,save_fig_path,n_class):
    #PR-AUCのマクロ平均を求める

    pr_auc_list = []
    fig,axes = plt.subplots(n_class,1,tight_layout=True,figsize=(6,12))
    plt.grid(True)

    for i in range(n_class):
        #クラスiが陽性、それ以外が陰性のときのPR曲線を求める。
        preds_cpy = copy.deepcopy(preds)
        #クラスiである確率だけ取り出す。
        preds_cpy = preds_cpy[:,i]

        labels_cpy = copy.deepcopy(labels)
        if labels_cpy.ndim == 1:
                labels_cpy = np.identity(n_class)[labels_cpy]
        labels_cpy = labels_cpy[:,i]

        precisions, recalls, thresholds = precision_recall_curve(labels_cpy, preds_cpy)
        precisions = np.insert(precisions,0,0)
        precisions = np.append(precisions,1)
        recalls = np.insert(recalls,0,1)
        recalls = np.append(recalls,0)

#        recalls = np.insert(recalls,0,0)
#        recalls = np.append(recalls,1)
#        precisions = np.insert(precisions,0,1)
#        precisions = np.append(precisions,0)


        try:
            pr_auc = auc(recalls, precisions)
        except:
            pr_auc = 0
        
        sns.set_style("whitegrid", {'grid.linestyle': '--'})
        sns.set_palette("Set3",desat=1.0)
        matplotlib.rcParams["font.family"] = 'IPAexGothic'
        sns.lineplot(x=precisions, y=recalls, ci=None,ax=axes[i],linewidth=3)
        #axes[i].plot(precisions,recalls,label = 'PR curve (area = %.3f'%pr_auc)
        #axes[i].set_title('PR curve Positive Class = ' + str(i+1))
        axes[i].set_xlim(-0.1,1.1)
        axes[i].set_ylim(-0.1,1.1)
        axes[i].grid(True)

        yticklabels = axes[i].get_yticklabels()
        axes[i].set_xticklabels([0.0,0.0,0.2,0.4,0.6,0.8,1.0],fontsize=18)
        axes[i].set_yticklabels([0.0,0.0,0.2,0.4,0.6,0.8,1.0],fontsize=18)
        axes[i].set_xlabel('Recall',fontsize=20)
        axes[i].set_ylabel('Precision',fontsize=20)
        
    fig.savefig(save_fig_path)

def make_PRBar(labels,preds,save_fig_path,n_class):
    #PR-AUCのマクロ平均を求める
    pr_auc_list = []
    #sns.set(font="IPAexGothic",font_scale=1.8)
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    sns.set_palette("Set3",desat=1.0)
    matplotlib.rcParams["font.family"] = 'IPAexGothic'
    fig,ax = plt.subplots(figsize=(8,8))

    for i in range(n_class):
        #クラスiが陽性、それ以外が陰性のときのPR曲線を求める。
        preds_cpy = copy.deepcopy(preds)
        #クラスiである確率だけ取り出す。
        preds_cpy = preds_cpy[:,i]

        labels_cpy = copy.deepcopy(labels)

        labels_cpy = labels_cpy[:,i]

        precisions, recalls, thresholds = precision_recall_curve(labels_cpy, preds_cpy)
        try:
            pr_auc = auc(recalls, precisions)
        except:
            pr_auc = 0
        
        pr_auc_list.append(pr_auc)
        
    ax.bar(list(range(n_class)),pr_auc_list,width=0.45)
    #ax.set_xticklabels(["","1","","2","","≧3"],fontsize=15)
    ax.set_yticklabels(["0","0.2","0.4","0.6","0.8","1.0"],fontsize=15)
    ax.set_title('陽性クラスに対するPR-AUC値',fontsize=20)
    ax.set_xlabel('陽性クラス',fontsize=18)
    ax.set_ylabel('PR-AUC値',fontsize=18)
    ax.set_ylim(0,1)
        
    fig.savefig(save_fig_path)
    print(save_fig_path)

def macro_pr_auc(labels,preds,n_class):
        # 2クラスだったら、PR-AUCをそのまま返す
        if n_class == 2:
            #preds = preds[:,1]

            try:
                precisions,recalls,thresholds = precision_recall_curve(labels, preds)
                pr_auc = auc(recalls, precisions)
            except:
                pr_auc = 0
            
            return pr_auc

        #3クラス以上の場合は、PR-AUCのマクロ平均を求める
        pr_auc_list = []
        for i in range(n_class):
            #クラスiが陽性、それ以外が陰性のときのPR曲線を求める。
            preds_cpy = copy.deepcopy(preds)
            preds_cpy = preds_cpy[:,i]

            labels_cpy = copy.deepcopy(labels)

            #1次元配列の時はラベルの値をワンホット表現に変換する
            if labels_cpy.ndim == 1:
                labels_cpy = np.identity(n_class)[labels_cpy]

            labels_cpy = labels_cpy[:,i]

            try:
                precisions, recalls, thresholds = precision_recall_curve(labels_cpy, preds_cpy)
                pr_auc = auc(recalls, precisions)
            except:
                pr_auc = 0
            pr_auc_list.append(pr_auc)
            
        print(pr_auc_list)    
        return statistics.mean(pr_auc_list)

def macro_f1(labels,preds,n_class):
        #PR-AUCのマクロ平均を求める
        f1_list = []
        for i in range(n_class):
            #クラスiが陽性、それ以外が陰性のときのPR曲線を求める。
            preds_cpy = copy.deepcopy(preds)

            #1次元配列の時はワンホット表現に変換する
            if preds_cpy.ndim == 1:
                preds_cpy = np.identity(n_class)[preds_cpy]
            preds_cpy = preds_cpy[:,i]

            labels_cpy = copy.deepcopy(labels)

            #1次元配列の時はラベルの値をワンホット表現に変換する
            if labels_cpy.ndim == 1:
                labels_cpy = np.identity(n_class)[labels_cpy]

            labels_cpy = labels_cpy[:,i]

            f1 = f1_score(labels_cpy, preds_cpy)
            f1_list.append(f1)
        print(f1_list)
        return statistics.mean(f1_list)

def make_F1Bar(labels,preds,save_fig_path,n_class):
    #PR-AUCのマクロ平均を求める
    f1_list = []
    #sns.set(font="IPAexGothic",font_scale=1.8)
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    sns.set_palette("Set3",desat=1.0)
    matplotlib.rcParams["font.family"] = 'IPAexGothic'
    fig,ax = plt.subplots(figsize=(8,8))

    for i in range(n_class):
        #クラスiが陽性、それ以外が陰性のときのPR曲線を求める。
        preds_cpy = copy.deepcopy(preds)
        #1次元配列の時はワンホット表現に変換する
        if preds_cpy.ndim == 1:
            preds_cpy = np.identity(n_class)[preds_cpy]
        preds_cpy = preds_cpy[:,i]

        labels_cpy = copy.deepcopy(labels)

        #1次元配列の時はラベルの値をワンホット表現に変換する
        if labels_cpy.ndim == 1:
            labels_cpy = np.identity(n_class)[labels_cpy]

        labels_cpy = labels_cpy[:,i]
        f1 = f1_score(labels_cpy, preds_cpy)
        f1_list.append(f1)
        
    ax.bar(list(range(n_class)),f1_list,width=0.45)
    #ax.set_xticklabels(["","1","","2","","≧3"],fontsize=15)
    ax.set_yticklabels(["0","0.2","0.4","0.6","0.8","1.0"],fontsize=15)
    ax.set_title('陽性クラスに対するF1値',fontsize=20)
    ax.set_xlabel('陽性クラス',fontsize=18)
    ax.set_ylabel('F1値',fontsize=18)
    ax.set_ylim(0,1)
        
    fig.savefig(save_fig_path)
