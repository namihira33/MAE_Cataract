import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import config
from PIL import Image
import os
import numpy as np
from utils import *
import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from glob import glob

#水平断面画像のデータセット
class OCThorizontalDatasetBase(Dataset):
    def __init__(self, root, image_list_file,transform=None,n_per_unit=1,d_type=5):
        image_names,labels,item_indexes = [],[],[]
        item_index = 0
        self.image_list_file = image_list_file

        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split(',')
                if isint(items[1]):

                    if d_type == 4:
                        gender = items[d_type][0]
                        label = 0 if gender == 'M' else 1

                    elif d_type == 3:
                        age = items[d_type]
                        label = 0 if (int(age)<60) else 1

                    else :
                        if isint(items[d_type]):                        
                            label = self.get_label(int(items[d_type][0]))
                    
                    for i in range(n_per_unit):
                        image_name = items[1] + '_' + items[2] + '_' + '{:0=3}'.format(int(i)) + '.jpg'
                        image_name = os.path.join(root,image_name)
                        image_names.append(image_name)
                        labels.append(label)
                        item_indexes.append(item_index)
                        item_index += 1

        self.image_names = np.array(image_names)
        self.labels = np.array(labels)
        self.item_indexes = np.array(item_indexes)
        self.transform = transform


    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('L')
        label = torch.eye(config.n_class)[self.labels[index]]
        #label = [self.labels[index]]
        #label = normal_distribution(self.labels[index])
        item_index = self.item_indexes[index]
        if self.transform is not None:
            image = self.transform(image)

        return (image,torch.Tensor(label),item_index) if self.image_list_file==config.train_info_list else (image,torch.Tensor(label),image_name)

    def __len__(self):
        return len(self.image_names)
        #return 1000

    def get_label(self, label_base):
        pass

    def pick_label(self, index):
        label = torch.eye(config.n_class)[self.labels[index]]
        return torch.Tensor(label)

        
class OCThorizontalDataset(OCThorizontalDatasetBase):
    def get_label(self, label_base):
        if label_base == 1:
            return 0
        elif label_base == 2:
            return 1
        elif label_base == 3:
            return 1
        else:
            return 2

#回転断面画像のデータセット
'''
class OCTspinDatasetBase(Dataset):
    def __init__(self, root, image_list_file,transform=None,n_per_unit=16,d_type=6):
        image_names,labels,item_indexes = [],[],[]
        item_index = 0
        self.image_list_file = image_list_file
        self.images = torch.empty(0)

        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split(',')
                if isint(items[1]):

                    if d_type == 4:
                        gender = items[d_type][0]
                        label = 0 if gender == 'M' else 1

                    elif d_type == 3:
                        age = items[d_type]
                        label = 0 if (int(age)<60) else 1

                    else :
                        if isint(items[d_type]):                        
                            label = self.get_label(int(items[d_type][0]))
                    
                    spin_images = torch.empty(0)

                    #ここで回転画像の追加
                    for i in range(n_per_unit):
                        image_name = items[1] + '_' + items[2] + '_' + '{:0=3}'.format(int(i)) + '.jpg'
                        image_name = os.path.join(root,image_name)
                        image_names.append(image_name)
                        item_indexes.append(item_index)
                        item_index += 1
                        labels.append(label)

        self.image_names = np.array(image_names)
        self.labels = np.array(labels)
        self.item_indexes = np.array(item_indexes)
        self.transform = transform
        print(len(self.image_names))

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = torch.eye(config.n_class)[self.labels[index]]
        item_index = self.item_indexes[index]
        if self.transform is not None:
            image = self.transform(image)

        return (image,torch.Tensor(label),item_index) if self.image_list_file==config.train_info_list else (image,torch.Tensor(label),image_name)

    def __len__(self):
        return len(self.image_names)

    def get_label(self, label_base):
        pass
'''

'''
class OCTspinDatasetBase(Dataset):
    def __init__(self, root, image_list_file,transform=None,n_per_unit=16,d_type=6):
        # まずはファイルを開いてID一覧を出力
        with open(image_list_file,"r") as f:
            lines = f.readlines()
        lines = [line.rstrip("\n") for line in lines] #右の改行文字の削除
        item_matrix = [line.split(',') for line in lines]

        self.transform = transform

        #画像の名前、ラベル、インデックスを記録する
        self.image_names = [os.path.join(root,item[1] + '_' + item[2] + '_' + '{:0=3}'.format(int(i)) + '.jpg')  for item in item_matrix for i in range(n_per_unit) if isint(item[1]) if isint(item[d_type])]
        self.images = [Image.open(os.path.join(root,item[1] + '_' + item[2] + '_' + '{:0=3}'.format(int(i)) + '.jpg')).convert('RGB') for item in item_matrix for i in range(n_per_unit) if isint(item[1]) if isint(item[d_type])]
        self.images = [self.transform(image) for image in self.images]
        self.labels = [self.get_label(int(item[d_type][0])) for item in item_matrix for i in range(n_per_unit) if isint(item[1]) if isint(item[d_type])]
        self.item_indexes = np.array(range(len(item_matrix)*n_per_unit))
        self.transform = transform
        self.image_list_file = image_list_file


    def __getitem__(self, index):
        image_name = self.image_names[index]
        #image = Image.open(image_name).convert('L')
        image = self.images[index]
        label = torch.eye(config.n_class)[self.labels[index]]
        item_index = self.item_indexes[index]
        #if self.transform is not None:
        #    image = self.transform(image)

        return (image,torch.Tensor(label),item_index) if self.image_list_file==config.train_info_list else (image,torch.Tensor(label),image_name)

    def __len__(self):
        return len(self.image_names)

    def pick_label(self, index):
        label = torch.eye(config.n_class)[self.labels[index]]
        return torch.Tensor(label)
'''

class OCTspinDatasetBase(Dataset):
    def __init__(self, root, image_list_file,transform=None,n_per_unit=16,d_type=6):
        # まずはファイルを開いてID一覧を出力
        with open(image_list_file,"r") as f:
            lines = f.readlines()
        lines = [line.rstrip("\n") for line in lines] #右の改行文字の削除
        item_matrix = [line.split(',') for line in lines]

        self.transform = transform
        self.images = torch.empty(0)

        #画像の名前、ラベル、インデックスを記録する
        self.image_names = [os.path.join(root,item[1] + '_' + item[2] + '_' + '{:0=3}'.format(int(0)) + '.jpg') for item in item_matrix if isint(item[1]) and isint(item[d_type]) ]
        labels = []

        #別々に扱う場合
        for item in item_matrix:
            spin_images = torch.empty(0)
            if isint(item[1]) and isint(item[d_type]):
                for i in range(n_per_unit):
                    image_name = item[1] + '_' + item[2] + '_' + '{:0=3}'.format(int(i)) + '.jpg'
                    subset_image = Image.open(os.path.join(root,image_name)).convert('L')
                    if transform is not None:
                        subset_image = self.transform(subset_image)
                    spin_images = torch.cat((spin_images,subset_image),0)
                #spin_images = self.transform(spin_images)
                spin_images = torch.reshape(spin_images,(1,n_per_unit,config.image_size,config.image_size))
                self.images = torch.cat((self.images,spin_images),0)
                label = self.get_label(int(item[d_type][0]))
                labels.append(label)
            print(len(labels))
            print(self.images.size())

        self.labels = labels
        print(len(self.labels))
        self.item_indexes = np.array(range(len(labels)))
        self.image_list_file = image_list_file


    def __getitem__(self, index):
        image_name = self.image_names[index]
        #image = Image.open(image_name).convert('L')
        image = self.images[index]
        label = torch.eye(config.n_class)[self.labels[index]]
        item_index = self.item_indexes[index]
        #if self.transform is not None:
        #    image = self.transform(image)

        return (image,torch.Tensor(label),item_index) if self.image_list_file==config.train_info_list else (image,torch.Tensor(label),image_name)

    def __len__(self):
        return len(self.labels)

    def pick_label(self, index):
        label = torch.eye(config.n_class)[self.labels[index]]
        return torch.Tensor(label)


        

# 症状なし:0 軽度:1 中度:2 重度:3と分類する。
class OCTspinDataset(OCTspinDatasetBase):
    def get_label(self, label_base):
        if label_base == 1:
            return 0
        elif label_base == 2:
            return 1
        elif label_base == 3:
            return 1
        else:
            return 2


class OCTspinMAEDatasetBase(Dataset):
    def __init__(self, root, image_list_file,transform=None,n_per_unit=16,d_type=6):

        self.transform = transform
        self.images = torch.empty(0)
        self.labels = []

        #画像の名前、ラベル、インデックスを記録する
        self.image_names = sorted(glob('../medicaldata/images/CASIA2_Add/train_add/*.png'))

        image_cnt = 0
        spin_images = torch.empty(0)

        '''

        for image_name in self.image_names:
            subset_image = Image.open(image_name).convert('L')
            if transform is not None:
                #subset_image = self.transform(subset_image)
                subset_image = transform(subset_image)
            print(image_name)
            spin_images = torch.cat((spin_images,subset_image),0)
            print(spin_images.size())
            image_cnt += 1

            if image_cnt == 16:
                spin_images = torch.reshape(spin_images,(1,n_per_unit,config.image_size,config.image_size))
                self.images = torch.cat((self.images,spin_images),0)
                label = 0
                self.labels.append(label)
                image_cnt = 0
                spin_images = torch.empty(0)
                print(self.images.size())
                '''


        #self.item_indexes = np.array(range(len(self.labels)*n_per_unit))

    def __getitem__(self, index):

        spin_image = torch.empty(0)
        
        image_names = self.image_names[index*16:(index+1)*16]

        for image_name in image_names:
            subset_image = Image.open(image_name).convert('L')
            if self.transform is not None:
                totensor = transforms.ToTensor()
                subset_image = totensor(subset_image)
                #mean_values = torch.mean(subset_image,dim=(1,2))
                #std_values = torch.std(subset_image,dim=(1,2))
                #normalize = transforms.Normalize(mean=mean_values.tolist(),std=std_values.tolist())
                #subset_image = normalize(subset_image)

            spin_image = torch.cat((spin_image,subset_image),0)

        if self.transform is not None:
            spin_image = self.transform(spin_image)

        image_name = self.image_names[index*16]

        #label = self.labels[index*16]
        #label = torch.eye(config.n_class)[self.labels[index]]
        #item_index = self.item_indexes[index]
        #if self.transform is not None:
        #    image = self.transform(image)

        return (spin_image,image_name)

    def __len__(self):
        return len(self.image_names) // 16

    #def pick_label(self, index):
    #    label = torch.eye(config.n_class)[self.labels[index]]
    #    return torch.Tensor(label)

        

# 症状なし:0 軽度:1 中度:2 重度:3と分類する。
class OCTspinMAEDataset(OCTspinMAEDatasetBase):
    def get_label(self, label_base):
        if label_base == 1:
            return 0
        else:
            return 1

def load_dataset(n_per_unit,d_type,preprocess,train_transform=None,root=None):

    valid_transform = \
            transforms.Compose([transforms.Resize(config.image_size),
                                transforms.CenterCrop(config.image_size),
                                transforms.ToTensor(),
                                #transforms.Normalize((0.5, ),(0.5, ))
                                ])
    test_transform = \
            transforms.Compose([transforms.Resize(config.image_size),
                                transforms.CenterCrop(config.image_size),
                                transforms.ToTensor(),
                                #transforms.Normalize((0.5, ),(0.5, ))
                                                    ])
                                                
        
    dataset = {}
    if n_per_unit == 1:
        if train_transform is None:
            train_transform = \
                transforms.Compose([
                                #transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                                #transforms.RandomHorizontalFlip(),
                                transforms.Resize(config.image_size),
                                transforms.CenterCrop(config.image_size),
                                transforms.ToTensor(),
                                #transforms.Normalize((0.5, ), (0.5, )),
                                ])
        dataset['train'] = \
            OCThorizontalDataset(root=os.path.join(config.data_root,preprocess),
                                    image_list_file=config.train_info_list,
                                    transform=train_transform,n_per_unit = n_per_unit,d_type=d_type)
        dataset['valid'] = \
            OCThorizontalDataset(root=os.path.join(config.data_root,preprocess),
                                    image_list_file=config.train_info_list,
                                    transform=valid_transform,n_per_unit = n_per_unit,d_type=d_type)                                    

        dataset['test'] = \
            OCThorizontalDataset(root=config.data_root,
                                    image_list_file=config.test_info_list,
                                    transform=test_transform,n_per_unit = n_per_unit,d_type=d_type)
    elif n_per_unit == 16:
        # MAEで使う場合だけ、train_transformを呼び出し時に入れる
        if train_transform is not None:
            dataset = \
                OCTspinMAEDataset(root=root,
                                        image_list_file=None,
                                        transform=train_transform,n_per_unit = n_per_unit,d_type=d_type)
            return dataset
        else:
            train_transform = \
                transforms.Compose([
                                #transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                                #transforms.RandomHorizontalFlip(),
                                transforms.Resize(config.image_size),
                                transforms.CenterCrop(config.image_size),
                                transforms.ToTensor(),
                                #transforms.Normalize((0.5, ), (0.5, )),
                                ])
            dataset['train'] = \
                OCTspinDataset(root=config.data_root,
                                        image_list_file=config.train_info_list,
                                        transform=train_transform,n_per_unit = n_per_unit,d_type=d_type)
        dataset['valid'] = \
            OCTspinDataset(root=config.data_root,
                                    image_list_file=config.train_info_list,
                                    transform=valid_transform,n_per_unit = n_per_unit,d_type=d_type)   
        dataset['test'] = \
            OCTspinDataset(root=config.data_root,
                                    image_list_file=config.test_info_list,
                                    transform=test_transform,n_per_unit = n_per_unit,d_type=d_type)


    return dataset

    '''
    CV実装のため、データセットのみの実装
    dataloader = {}
    
    dataloader['train'] = \
        torch.utils.data.DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    num_workers=0)
    dataloader['test'] = \
        torch.utils.data.DataLoader(test_dataset,
                                    batch_size=batch_size,
                                    num_workers=0)
    return dataloader
    '''


    '''
    CV実装のため、データセットのみの実装
    dataloader = {}
    
    dataloader['train'] = \
        torch.utils.data.DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    num_workers=0)
    dataloader['test'] = \
        torch.utils.data.DataLoader(test_dataset,
                                    batch_size=batch_size,
                                    num_workers=0)
    return dataloader
    '''

