import os
import argparse
import json
import glob
import shutil
import random
import numpy as np
import tqdm
import cv2
import math
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity

name = ['(', ')', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'B', 'C', 'M', 'N', 'S', 'T', 'Y', 'A', 'D',
        'E', 'H', 'L', 'W', '/', 'x', 'P', 'F', 'G', 'X', '.', 'o', 'J', 'U', 'I', 'K', 'Q', 'R', 'V', 'Z']


def gen_rec_label(input_dir, out_label, imgs_dir='img/'):
    with open(out_label, 'w') as out_file:
        for input_label in os.listdir(input_dir):
            img_path = imgs_dir + input_label.replace('.txt', '.jpg')
            with open(os.path.join(input_dir, input_label), 'r') as f:
                content = []
                for line in f.readlines():
                    tmp = line.strip().strip('\n').split(' ')
                    content.append(name[int(tmp[0])])
                out_file.write(img_path + '\t' + ''.join(content) + '\n')


def reverse_cvt(size, box):
    x = box[0] * size[0]
    y = box[1] * size[1]
    w = box[2] * size[0]
    h = box[3] * size[1]
    return [(x + 1) - w / 2, (y + 1) - h / 2], [(x + 1) + w / 2, (y + 1) - h / 2], \
           [(x + 1) + w / 2, (y + 1) + h / 2], [(x + 1) - w / 2, (y + 1) + h / 2]


def yolo_to_icdar(input_label, input_image, output_label):
    img = cv2.imread(input_image)
    h, w = img.shape[:2]
    with open(output_label, 'w') as oup:
        times = 0
        rows = []
        with open(input_label, 'r') as inp:
            for line in inp.readlines():
                row = line.strip().replace('\n', '').split(' ')
                rev = reverse_cvt((w, h), (float(row[1]), float(row[2]), float(row[3]), float(row[4])))
                rows.append([[float(row[0]), 0], rev[0], rev[1], rev[2], rev[3]])
        rows_arr = np.asarray(rows)
        mixed = np.array(rows_arr[:, 1, 1])
        # rows_arr = np.array(rows)
        # mixed = np.asarray([value[1][0][1] for value in rows])
        range_mix = np.linspace(np.min(mixed) - 3, np.max(mixed) + 3)
        kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(mixed.reshape((-1, 1)))  # 构造核密度估计
        samples = kde.score_samples(range_mix.reshape(-1, 1))  # 在区间上用核函数拟合
        mi = argrelextrema(samples, np.less)[0]  # 计算相对极值
        if len(mi) > 0:
            temp = rows_arr[rows_arr[:, 1, 1] < range_mix[mi][times]].tolist()
            temp.sort(key=lambda x: x[1][0])
            text = ''.join([name[int(word[0][0])] for word in temp])
            temp = np.asarray(temp, dtype=np.int32)
            ymin = np.min(temp[:, 1, 1])
            ymax = np.max(temp[:, 3, 1])
            oup.write(str(temp[0, 1, 0]) + ',' + str(ymin) + ',')
            oup.write(str(temp[-1, 2, 0]) + ',' + str(ymin) + ',')
            oup.write(str(temp[-1, 2, 0]) + ',' + str(ymax) + ',')
            oup.write(str(temp[0, 1, 0]) + ',' + str(ymax) + ',' + text + '\n')
            while True:
                if times == len(mi) - 1:
                    break
                temp = rows_arr[(rows_arr[:, 1, 1] >= range_mix[mi][times]) *
                                (rows_arr[:, 1, 1] < range_mix[mi][times + 1])].tolist()
                temp.sort(key=lambda x: x[1][0])
                text = ''.join([name[int(word[0][0])] for word in temp])
                temp = np.asarray(temp, dtype=np.int32)
                xmin = temp[0, 1, 0]
                xmax = temp[-1, 2, 0]
                ymin = np.min(temp[:, 1, 1])
                ymax = np.max(temp[:, 3, 1])
                oup.write(','.join([str(xmin), str(ymin), str(xmax), str(ymin),
                                    str(xmax), str(ymax), str(xmin), str(ymax), text]) + '\n')
                times += 1
            temp = rows_arr[rows_arr[:, 1, 1] >= range_mix[mi][times]].tolist()
            temp.sort(key=lambda x: x[1][0])
            text = ''.join([name[int(word[0][0])] for word in temp])
            temp = np.asarray(temp, dtype=np.int32)
            ymin = np.min(temp[:, 1, 1])
            ymax = np.max(temp[:, 3, 1])
            oup.write(str(temp[0, 1, 0]) + ',' + str(ymin) + ',')
            oup.write(str(temp[-1, 2, 0]) + ',' + str(ymin) + ',')
            oup.write(str(temp[-1, 2, 0]) + ',' + str(ymax) + ',')
            oup.write(str(temp[0, 1, 0]) + ',' + str(ymax) + ',' + text + '\n')
        else:
            temp = rows_arr.tolist()
            temp.sort(key=lambda x: x[1][0])
            text = ''.join([name[int(word[0][0])] for word in temp])
            temp = np.asarray(temp, dtype=np.int32)
            ymin = np.min(temp[:, 1, 1])
            ymax = np.max(temp[:, 3, 1])
            oup.write(str(temp[0, 1, 0]) + ',' + str(ymin) + ',')
            oup.write(str(temp[-1, 2, 0]) + ',' + str(ymin) + ',')
            oup.write(str(temp[-1, 2, 0]) + ',' + str(ymax) + ',')
            oup.write(str(temp[0, 1, 0]) + ',' + str(ymax) + ',' + text + '\n')


def gen_det_label(input_dir, out_label, imgs_dir="img/"):
    with open(out_label, 'w') as out_file:
        for label_file in os.listdir(input_dir):
            # img_path = root_path + label_file[3:-4] + ".jpg"
            img_path = imgs_dir + label_file[0:-4] + ".jpg"
            label = []
            with open(os.path.join(input_dir, label_file), 'r', encoding='UTF-8') as f:
                for line in f.readlines():
                    tmp = line.strip("\n\r").replace("\xef\xbb\xbf",
                                                     "").split(',')
                    points = tmp[:8]
                    s = []
                    for i in range(0, len(points), 2):
                        b = points[i:i + 2]
                        b = [int(t) for t in b]
                        s.append(b)
                    result = {"transcription": tmp[8], "points": s}
                    label.append(result)

            out_file.write(img_path + '\t' + json.dumps(
                label, ensure_ascii=False) + '\n')


def rec(data_dir="yolo_rec_data", output_dir="output_rec", split_percent=None):
    # data_dir = "yolo_rec_data"  # 原始数据集
    # output_dir = "output_rec"  # 生成数据集
    # split_percent = 0.2  # 评估数据占比
    train_data_dir = os.path.join(output_dir, 'train')
    train_imgs = os.path.join(train_data_dir, 'img')
    train_temp = os.path.join(train_data_dir, 'temp')
    test_data_dir = os.path.join(output_dir, 'test')
    test_imgs = os.path.join(test_data_dir, 'img')
    test_temp = os.path.join(test_data_dir, 'temp')
    # labels_temp = data_dir + '_temp'  # 暂时存放原标签

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(train_data_dir):
        os.mkdir(train_data_dir)
        os.mkdir(train_imgs)
        os.mkdir(train_temp)
    if not os.path.exists(test_data_dir):
        os.mkdir(test_data_dir)
        os.mkdir(test_imgs)
        os.mkdir(test_temp)

    print("Generate rec label for testing...")
    labels = glob.glob(os.path.join(data_dir, '*.txt'))
    random.shuffle(labels)
    if split_percent is None:
        percent = 0.2
    else:
        percent = float(split_percent)
    test_data_len = math.floor(percent * len(labels))
    for i in tqdm.tqdm(range(test_data_len)):
        label = os.path.join(test_temp, os.path.split(labels[i])[1])
        image = os.path.join(test_imgs, os.path.split(labels[i])[1].replace('.txt', '.jpg'))
        shutil.copy(labels[i].replace('.txt', '.jpg'), image)
        shutil.copy(labels[i], label)
        # shutil.move(os.path.join(data_dir, image), os.path.join(labels_temp, image))
        # shutil.move(labels[i], os.path.join(labels_temp, os.path.split(labels[i])[1]))
    gen_rec_label(test_temp, os.path.join(test_data_dir, 'test.txt'))

    print("\nGenerate rec label for training...")
    # labels = glob.glob(os.path.join(data_dir, '*.txt'))
    for i in tqdm.tqdm(range(math.ceil((1 - percent) * len(labels)))):
        j = i + test_data_len
        label = os.path.join(train_temp, os.path.split(labels[j])[1])
        image = os.path.join(train_imgs, os.path.split(labels[j])[1].replace('.txt', '.jpg'))
        shutil.copy(labels[j].replace('.txt', '.jpg'), image)
        shutil.copy(labels[j], label)
        # shutil.move(labels[i], os.path.join(labels_temp, os.path.split(labels[i])[1]))
    gen_rec_label(train_temp, os.path.join(train_data_dir, 'train.txt'))

    print("\nGenerate dict file for configuring...")
    with open(os.path.join(output_dir, 'my_dict.txt'), 'w') as dic:
        for i in range(len(name)):
            dic.write(name[i] + '\n')


def det(data_dir="yolo_det_data", output_dir="output_det", split_percent=None):
    # data_dir = "yolo_det_data"  # 原始数据集
    # output_dir = "output_det"  # 生成数据集
    # split_percent = 0.2  # 评估数据占比
    train_data_dir = os.path.join(output_dir, 'train')
    train_imgs = os.path.join(train_data_dir, 'img')
    train_temp = os.path.join(train_data_dir, 'temp')
    test_data_dir = os.path.join(output_dir, 'test')
    test_imgs = os.path.join(test_data_dir, 'img')
    test_temp = os.path.join(test_data_dir, 'temp')
    # labels_temp = data_dir + '_temp'  # 暂时存放原标签

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(train_data_dir):
        os.mkdir(train_data_dir)
        os.mkdir(train_imgs)
        os.mkdir(train_temp)
    if not os.path.exists(test_data_dir):
        os.mkdir(test_data_dir)
        os.mkdir(test_imgs)
        os.mkdir(test_temp)
    # if not os.path.exists(labels_temp):
    #     os.mkdir(labels_temp)

    print("Generate det label for testing...")
    labels = glob.glob(os.path.join(data_dir, '*.txt'))
    random.shuffle(labels)
    if split_percent is None:
        percent = 0.2
    else:
        percent = float(split_percent)
    test_data_len = math.floor(percent * len(labels))
    for i in tqdm.tqdm(range(test_data_len)):
        label = os.path.join(test_temp, os.path.split(labels[i])[1])
        image = os.path.join(test_imgs, os.path.split(labels[i])[1].replace('.txt', '.jpg'))
        shutil.copy(labels[i].replace('.txt', '.jpg'), image)
        yolo_to_icdar(labels[i], image, label)
        # shutil.move(os.path.join(data_dir, image), os.path.join(labels_temp, image))
        # shutil.move(labels[i], os.path.join(labels_temp, os.path.split(labels[i])[1]))
    gen_det_label(test_temp, os.path.join(test_data_dir, 'test.txt'))

    print("\nGenerate det label for training...")
    # labels = glob.glob(os.path.join(data_dir, '*.txt'))
    for i in tqdm.tqdm(range(math.ceil((1 - percent) * len(labels)))):
        j = i + test_data_len
        label = os.path.join(train_temp, os.path.split(labels[j])[1])
        image = os.path.join(train_imgs, os.path.split(labels[j])[1].replace('.txt', '.jpg'))
        shutil.copy(labels[j].replace('.txt', '.jpg'), image)
        yolo_to_icdar(labels[j], image, label)
        # shutil.move(labels[i], os.path.join(labels_temp, os.path.split(labels[i])[1]))
    gen_det_label(train_temp, os.path.join(train_data_dir, 'train.txt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        type=str,
        default="rec",
        help='Generate rec_label or det_label, can be set rec or det')
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        default=".",
        help='The input directory of images and labels')
    parser.add_argument(
        '--output_dir',
        type=str,
        default="./output",
        help='The output directory of images and labels')
    parser.add_argument(
        '--split_percent',
        type=str,
        default=None,
        help='The percent of the test data in the whole input data set')

    args = parser.parse_args()
    if args.mode == "rec":
        rec(args.input_dir, args.output_dir, args.split_percent)
    elif args.mode == "det":
        det(args.input_dir, args.output_dir, args.split_percent)
