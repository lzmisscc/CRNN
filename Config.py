import alphabets

raw_folder = ''
train_data = './CRNN_DATA/train_lmdb'
test_data = './CRNN_DATA/test_lmdb'
random_sample = True
random_seed = 1111
using_cuda = True
keep_ratio = True
gpu_id = '0'
model_dir = './model_h5'
data_worker = 10
batch_size = 16
img_height = 32
img_width = 160
alphabet = alphabets.alphabet
epoch = 1600
# 调小display_interval就可正常显示
display_interval = 60
# 以下无效
save_interval = 40
test_interval = 40
# 以上无效
test_disp = 20
test_batch_num = 64
lr = 0.0001
beta1 = 0.5
infer_img_w = 160
