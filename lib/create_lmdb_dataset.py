import lmdb
import cv2
import numpy as np
import os

OUT_PATH = '/home/lz/PycharmProjects/OCR/recognizer/crnn/CRNN_DATA/test_lmdb/'
IN_PATH = "/home/lz/cd_data/cropline2.txt"
PREFIX = "/home/lz/cd_data/cropline2"


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    try:
        imageBuf = np.fromstring(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
    except:
        return False
    else:
        if imgH * imgW == 0:
            return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert (len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = os.path.join(PREFIX, imagePathList[i]).split()[0].replace('\n', '').replace('\r\n', '')
        print(imagePath)
        label = ''.join(labelList[i])
        print(label)
        # if not os.path.exists(imagePath):
        #     print('%s does not exist' % imagePath)
        #     continue

        with open(imagePath, 'r') as f:
            imageBin = f.read()

        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue
        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
        print(cnt)
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def gen_lmdb():
    outputPath = OUT_PATH
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    imgdata = open(IN_PATH)
    imagePathList = list(imgdata)

    labelList = []
    for line in imagePathList:
        word = line.split(" ", 1)[1].replace(' ', '')
        # word = "".join(line.split(" ", 1)[1].split(" "))
        # print("word", word)
        labelList.append(word.strip('\n').replace('\r', '').replace('\t', ''))
    createDataset(outputPath, imagePathList, labelList)


if __name__ == '__main__':
    gen_lmdb()
    OUT_PATH = '/home/lz/PycharmProjects/OCR/recognizer/crnn/CRNN_DATA/train_lmdb/'
    IN_PATH = "/home/lz/cd_data/cropline1.txt"
    PREFIX = "/home/lz/cd_data/cropline1"
    gen_lmdb()
