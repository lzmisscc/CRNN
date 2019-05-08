import time
import torch
import os
from torch.autograd import Variable
import lib.convert
import lib.dataset
from PIL import Image
import Net.net as Net
import alphabets
import sys
import Config

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

crnn_model_path = './model_h5/netCRNN_3190_180.pth'
IMG_ROOT = '/home/lz/cd_data/cropline2/'
GT_TXT = '/home/lz/cd_data/cropline2.txt'
running_mode = 'gpu'
alphabet = alphabets.alphabet
nclass = len(alphabet) + 1


def crnn_recognition(cropped_image, model):
    converter = lib.convert.strLabelConverter(alphabet)

    image = cropped_image.convert('L')

    ### Testing images are scaled to have height 32. Widths are
    # proportionally scaled with heights, but at least 100 pixels
    w = int(image.size[0] / (280 * 1.0 / Config.infer_img_w))
    # scale = image.size[1] * 1.0 / Config.img_height
    # w = int(image.size[0] / scale)

    transformer = lib.dataset.resizeNormalize((w, Config.img_height))
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('results: {0}'.format(sim_pred))


def read_gt(gt_txt_name):
    with open(gt_txt_name, "r") as f:
        x = f.readline()
        L = []
        while x:
            y = x.strip("\n").split(" ", 1)
            L.append(y)
            x = f.readline()
    return L


if __name__ == '__main__':

    # crnn network
    model = Net.CRNN(nclass)
    if running_mode == 'gpu' and torch.cuda.is_available():
        model = model.cuda()
        model.load_state_dict(torch.load(crnn_model_path))
    else:
        model.load_state_dict(torch.load(crnn_model_path, map_location='cpu'))

    print('loading pretrained model from {0}'.format(crnn_model_path))

    # files = sorted(os.listdir(IMG_ROOT))
    for file in read_gt(GT_TXT):
        started = time.time()
        full_path = os.path.join(IMG_ROOT, file[0])
        print("=============================================")
        print("ocr image is %s" % full_path)
        image = Image.open(full_path)

        crnn_recognition(image, model)
        print(file[1])
        finished = time.time()
        print('elapsed time: {0}'.format(finished - started))
