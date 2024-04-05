data_root = '../medicaldata/images/CASIA2_16'

# 先生方によるラベル付与の分割データ
#train_info_list = '../medicaldata/txt/as_oct_train_preprocess.csv'
#test_info_list = '../medicaldata/txt/as_oct_test.csv'

# 12月発表時に使用したデータ
train_info_list = '../medicaldata/txt/casia16_train_list.csv'
test_info_list = '../medicaldata/txt/casia16_test_list.csv'

#pickleの位置
#normal_pkl = '../medicaldata/pkls/OCT_ViT_spin.pkl'
#normal_pkl = '../medicaldata/pkls/OCT_ViT_horizontal_N3.pkl'
#normal_pkl = '../medicaldata/pkls/OCT_ViT_horizontal_N3_DA.pkl'
normal_pkl = '../medicaldata/pkls/OCT_ViT_spin_N4.pkl'
#normal_pkl = '../medicaldata/pkls/OCT_C2_16to1.pkl'
#normal_pkl = '../medicaldata/pkls/OCT_C2_16to1spin.pkl'
#normal_pkl = '../medicaldata/pkls/OCT_P2_16to1spin.pkl'
#normal_pkl = '../medicaldata/pkls/OCT_C2_horizontal.pkl'
#normal_pkl = 'OCT_MAE2_transforms2.pkl'
#normal_pkl =  '../medicaldata/pkls/OCT_N3_spin16to1.pkl'
#normal_pkl =  '../medicaldata/pkls/OCT_Over_N3_spin16to1.pkl'


#MAE_dataset_pkl = '../medicaldata/pkls/OCT_MAEViT_spin2.pkl'
#MAE_dataset_pkl = '../medicaldata/pkls/OCT_MAEViT_C2_16to1_0.8.pkl'

#mae_path = './model/MAE_800ep.pth'
mae_path = './model/MAE1ch_800ep.pth'

MODEL_DIR_PATH = './model/'
LOG_DIR_PATH = './log/'
n_per_unit = 1
image_size = 224
n_class = 4

# train_info_list = '../medicaldata/txt/casia16_train_list.csv'
# test_info_list = '../medicaldata/txt/casia16_test_list.csv'