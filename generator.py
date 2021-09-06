import os
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import random
import numpy as np
from matplotlib import cm
import cv2
from tqdm import tqdm

from util import *


class ImageGenerator():

    # def __init__(self, font_dir, output_dir, font_size, text_corpus_path, date_corpus_path,
    #              bg_path, img_size):
    def __init__(self, font_dir, output_dir, font_size_min, font_size_max, text_corpus_path, date_corpus_path,
                 bg_path, img_size):
        # def __init__(self, font_dir, output_dir, font_size_min, font_size_max, text_corpus_path, date_corpus_path,
        # img_size=(1280, 960), bg_colour=(255, 255, 255, 255)):
        self.img_size = img_size
        # self.bg_colour = bg_colour
        self.bg = bg_path
        self.output_dir = output_dir
        self.font_dir = font_dir
        self.text_corpus = load_dict(text_corpus_path)
        self.date_corpus = load_date_dict(date_corpus_path)
        self.fonts = load_font(self.font_dir)
        self.font = self.fonts[random.choice(range(len(self.fonts)))]
        self.noise_types = load_noise_type()
        self.corpus = [self.text_corpus, self.date_corpus]
        # self.font_size = font_size
        # self.start_x = random.randrange(5, 25)
        # self.start_y = random.randrange(5, 25)
        self.font_size_min = font_size_min
        self.font_size_max = font_size_max

    def draw_text(self, draw, x, y, text, file, font_size, fill=(0,0,0), draw_rectangle=False):

        # print(os.path.join(self.font_dir,self.fonts[random.randrange(0,len(self.fonts))]))
        # font = ImageFont.truetype(os.path.join(self.font_dir, self.fonts[random.randrange(0, len(self.fonts))]),
        font = ImageFont.truetype(os.path.join(self.font_dir, self.font),
                                  font_size)
        # random.randrange(self.font_size_min, self.font_size_max))

        (w, h) = textsize(text=text, font=font)
        if x + w > (self.img_size[0] - 20) or y + h > (self.img_size[1] - 20):
            # if x + w > self.img_size[0] or y + h > self.img_size[1]:
            return draw, (w, h), False
        draw.text((x, y), text, fill=fill, font=font)
        file.write("{},{},{},{},{},{},{},{},{}\n".format(x, y, x + w, y, x + w, y + h, x, y + h, text))

        # (w,h) = draw.textsize(text)
        if draw_rectangle:
            draw.rectangle([x - 1, y + 2, x + w, y + h], outline="black")

        return draw, (w, h), True

    def generateImages(self, gen_num, change_num, line_num_min, line_num_max, draw_rectangle=True, rows_num=None):

        if rows_num is None:
            rows_num = [6, 20]
        for i in tqdm(range(gen_num)):

            # file = open(os.path.join(self.output_dir, "img_" + str(i) + ".txt"), 'w')
            file = open(os.path.join(self.output_dir, "{:0>4d}".format(change_num + i) + ".txt"), 'w')
            # im = Image.new('RGB', self.img_size, self.bg_colour)
            ###################################small work around as noise code uses cv2 not PIL########################################################
            # im.save("temp.jpg")
            # cv2.imwrite("temp.png",
            #             noisy(self.noise_types[random.randrange(0, len(self.noise_types))], cv2.imread("temp.jpg")))
            ############################################################################################################################################
            # im = Image.open("temp.png")
            im = Image.open(self.bg)  # 字体背景
            draw = ImageDraw.Draw(im)
            start_x = random.randrange(50, 650)  # 字体位置
            start_y = random.randrange(0, 50) + 340
            # start_x = random.randrange(5, 25)
            # start_y = random.randrange(5, 25)
            color = random.randrange(0, 120)  # 字体颜色
            font_size = random.randrange(self.font_size_min, self.font_size_max)  # 字体大小
            fill = (color, color, color)
            # y_list = []
            # y_list.append(start_y)
            # line_number_end = random.randrange(2, 25)
            line_number_end = random.randrange(line_num_min, line_num_max)
            line_number = 1

            # while True:
            #
            #     # corpus_selection = self.corpus[random.randrange(0, len(self.corpus))]
            #     draw, shape, flag = self.draw_text(draw, start_x, start_y,
            #                                        # corpus_selection[random.randrange(0, len(corpus_selection))], file,
            #                                        self.text_corpus[random.randrange(0, len(self.text_corpus))], file,
            #                                        draw_rectangle)
            #
            #     start_x = start_x + shape[0] + 10
            #     y_list.append(start_y + shape[1])
            #     # if start_x > 1280 or flag == False:
            #     if start_x > self.img_size[0] or flag == False:
            #         break

            draw, shape, flag = self.draw_text(draw, start_x, start_y,
                                               # corpus_selection[random.randrange(0, len(corpus_selection))], file,
                                               self.text_corpus[random.randrange(0, len(self.text_corpus))], file,
                                               font_size, fill, draw_rectangle)

            while line_number < line_number_end:
                start_x = random.randrange(50, 650)
                start_y = start_y + shape[1] + random.randrange(0, 10)

                draw, shape, flag = self.draw_text(draw, start_x, start_y,
                                                   # corpus_selection[random.randrange(0, len(corpus_selection))],
                                                   self.text_corpus[random.randrange(0, len(self.text_corpus))], file,
                                                   font_size, fill, draw_rectangle)

                line_number += 1


            # while True:
            #     if line_number < line_number_end:
            #         start_x = random.randrange(5, 25)
            #         start_y = max(y_list) + 10
            #         y_list = []
            #         y_list.append(start_y)
            #
            #         while True:
            #             # corpus_selection = self.corpus[random.randrange(0, len(self.corpus))]
            #             draw, shape, flag = self.draw_text(draw, start_x, start_y,
            #                                                # corpus_selection[random.randrange(0, len(corpus_selection))],
            #                                                self.text_corpus[random.randrange(0, len(self.text_corpus))],
            #                                                file, draw_rectangle)
            #             start_x = start_x + shape[0] + 10
            #             y_list.append(start_y + shape[1])
            #             # if start_x > 1260 or flag == False:
            #             if start_x > self.img_size[0] - 20 or flag == False:
            #                 break
            #
            #         # if start_y + 10 > 700:
            #         if start_y + 10 > self.img_size[1] - 60:
            #             line_number += 1
            #             break
            #         line_number += 1
            #     else:
            #         break

            im.save(os.path.join(self.output_dir, "{:0>4d}".format(change_num + i) + ".jpg"))
            im.close()
            # os.remove("temp.jpg")
            # os.remove("temp.png")
            file.close()
