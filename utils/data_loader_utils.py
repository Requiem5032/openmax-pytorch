import cv2
import numpy as np


def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def make_square(img):
    if img.shape[0] > img.shape[1]:
        img = np.rollaxis(img, 1, 0)
    toppadlen = (img.shape[1] - img.shape[0])//2
    bottompadlen = img.shape[1] - img.shape[0] - toppadlen
    toppad = img[:5, :, :].mean(0, keepdims=True).astype(img.dtype)
    toppad = np.repeat(toppad, toppadlen, 0)
    bottompad = img[-5:, :, :].mean(0, keepdims=True).astype(img.dtype)
    bottompad = np.repeat(bottompad, bottompadlen, 0)
    return np.concatenate((toppad, img, bottompad), axis=0)


def mixup_loader(idx, df, dataset, colors):
    mixid = df.sample()
    # if dataset=="train":
    #     print(mixid)
    ratio = np.random.rand()

    targets1 = df.loc[idx, 'label']
    targets2 = mixid['label'].values[0]
    # print("Target1, ", targets1, type(targets1), dataset)
    # print("Target2, ", targets2, type(targets2), dataset)
    targets = ratio*targets1 + (1-ratio)*targets2

    image1 = load_image(df.loc[idx, 'Id'], dataset, colors)
    image2 = load_image(mixid['Id'].values[0], dataset, colors)
    image = (ratio*image1 + (1-ratio)*image2).round().astype('uint8')
    # print("ids = {}, {}. Ratio = {}".format(df.loc[idx, 'Id'], mixid[0], ratio))
    return image, targets
