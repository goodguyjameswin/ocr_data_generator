import cv2
import numpy as np
import os
import random
import math
from PIL import Image


def show_image(img_path, label_path):
    """
    标签可视化
    输入:
        img(array): 生成图像
        label: 图像的标签信息
    """
    img = cv2.imread(img_path)
    with open(label_path, 'r') as lbl:
        for line in lbl.readlines():
            box = line.strip().split(' ')
            cv2.line(img, (int(float(box[:-1:2][0])), int(float(box[1:-1:2][0]))),
                     (int(float(box[:-1:2][1])), int(float(box[1:-1:2][1]))), (0, 255, 0), 3)
            cv2.line(img, (int(float(box[:-1:2][1])), int(float(box[1:-1:2][1]))),
                     (int(float(box[:-1:2][2])), int(float(box[1:-1:2][2]))), (0, 255, 0), 3)
            cv2.line(img, (int(float(box[:-1:2][2])), int(float(box[1:-1:2][2]))),
                     (int(float(box[:-1:2][3])), int(float(box[1:-1:2][3]))), (0, 255, 0), 3)
            cv2.line(img, (int(float(box[:-1:2][3])), int(float(box[1:-1:2][3]))),
                     (int(float(box[:-1:2][0])), int(float(box[1:-1:2][0]))), (0, 255, 0), 3)
            # 类别标签默认在原模板图的左上角显示
            cv2.putText(img, box[-1], (int(float(box[0])), int(float(box[1]))),
                        cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.5, (255, 0, 0), 1)
    # 设置显示窗口
    cv2.namedWindow('image', 0)  # 1表示原图
    cv2.moveWindow('image', 1200, 600)
    cv2.resizeWindow('image', 460, 460)  # 可视化的图片大小
    cv2.imshow('image', img)
    # cv2.imwrite("./test_result/0001" + ".jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class DataMaker:
    def __init__(self, bg_dir, tpl_dir, sv_img_dir="./images", sv_lbl_dir="./labels"):
        """
        图像增广
        参数:
            bg_dir: 背景图片文件夹
            tpl_dir: 模板图片文件夹（图片格式为png），文件夹名作为模板的类别
            sv_img_dir: 存储生成图片的文件夹（默认为当前目录下的images文件夹）
            sv_lbl_dir: 存储标签文件的文件夹（默认为当前目录下的labels文件夹）
        """
        self.bg_dir = bg_dir
        self.tpl_dir = tpl_dir
        self.sv_img_dir = sv_img_dir
        self.sv_lbl_dir = sv_lbl_dir
        if not os.path.exists(self.sv_img_dir):
            os.mkdir(self.sv_img_dir)
        if not os.path.exists(self.sv_lbl_dir):
            os.mkdir(self.sv_lbl_dir)

        # 读取背景文件
        counts = len(os.listdir(bg_dir))
        assert counts > 0, '未读取到有效背景图！'
        self.bg_path = None
        self.roi = None

        # 读取模板文件
        counts = len(os.listdir(tpl_dir))
        assert counts > 0, '未读取到有效模板图！'
        self.tpl_path = [os.path.join(tpl_dir, tpl_img) for tpl_img in os.listdir(tpl_dir)]
        # if counts > 1:
        #     self.bg_path = [os.path.join(bg_dir, bg_img) for bg_img in os.listdir(bg_dir)]
        #     print("依次选择贴图区域！")
        #     self.roi = [cv2.selectROI('roi', cv2.imread(bg_path, 1), False) for bg_path in self.bg_path]

    @classmethod
    def rotate_xy(cls, x, y, angle, cx, cy):
        """
        点(x,y) 绕(cx,cy)点旋转 angle（单位是弧度）得到新的坐标
        """
        x_new = (x - cx) * math.cos(angle) - (y - cy) * math.sin(angle) + cx
        y_new = (x - cx) * math.sin(angle) + (y - cy) * math.cos(angle) + cy
        return x_new, y_new

    @classmethod
    def compute_intersection(cls, set_1, set_2):
        """
        计算box集合之间的交并比
        输入:
            set_1: shape=(n1, 4), box(xmin, ymin, xmax, ymax)
            set_2: shape=(n2, 4), box(xmin, ymin, xmax, ymax)
        输出:
            set_1中每个box与set_2中每个box之间的Jaccard系数（iou），shape=(n1, n2)
        """
        lower_bounds = np.maximum(np.expand_dims(set_1[:, :2], axis=1),
                                  np.expand_dims(set_2[:, :2], axis=0))  # (n1, n2, 2)
        upper_bounds = np.minimum(np.expand_dims(set_1[:, 2:], axis=1),
                                  np.expand_dims(set_2[:, 2:], axis=0))  # (n1, n2, 2)
        intersection_dims = np.clip(upper_bounds - lower_bounds, 0, np.inf)
        # 计算交集
        intersection = intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)
        # 计算每个集合中box的面积
        area_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
        area_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)
        # 计算并集
        union = np.expand_dims(area_set_1, axis=1) + \
                np.expand_dims(area_set_2, axis=0) - intersection  # (n1, n2)
        return intersection / union  # (n1, n2)

    @classmethod
    def rotate_image(cls, img, angle=0., scale=1.):
        """
        对图片做旋转填充，记录原图的位置信息
        输入:
            img: 图像
            angle: 旋转角度（为统一标准，规定角度为顺时针旋转0至360度）
            scale: 变换尺度（默认1.0）
        输出:
            rot_img: 旋转填充后的图像
            array: 模板位置信息，shape=(4, 2)，从左上角开始沿顺时针方向的各顶点坐标
        """
        w = img.shape[1]
        h = img.shape[0]
        # 角度单位转弧度
        rangle = np.deg2rad(angle)
        # 计算旋转填充后的图片宽高
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        # 获取旋转矩阵
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), -angle, scale)  # 为统一角度，输入参数angle取负值
        # 旋转并填充，原图片应位于背景图的中央
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 做仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)),
                                                int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
        # 获取原始模板图的四个顶点坐标，然后转换到填充后的坐标系下
        tl = np.dot(rot_mat, np.array([0., 0., 1]))
        tr = np.dot(rot_mat, np.array([w, 0., 1]))
        br = np.dot(rot_mat, np.array([w, h, 1]))
        bl = np.dot(rot_mat, np.array([0., h, 1]))
        return rot_img, np.array([tl, tr, br, bl])

    @classmethod
    def to_transparent(cls, img):
        """
        对图片的单色背景做透明处理
        输入：
            img: 待处理图片（pillow格式）
        输出：
            img: 已处理图片（分黑色背景和白色背景两种情况，pillow格式）
        """
        img = img.convert("RGBA")
        pixel_data = img.load()
        for h in range(img.height):
            for w in range(img.width):
                pixel = pixel_data[w, h]
                r = pixel[0]
                g = pixel[1]
                b = pixel[2]
                # 四通道，色彩值小于淡黑色，则将像素点变为透明块
                if r < 10 and g < 10 and b < 10:
                    pixel_data[w, h] = (255, 255, 255, 0)
                # 四通道，色彩值大于浅灰色，则将像素点变为透明快
                # if r > 240 and g > 240 and b > 240:
                #     pixel_data[w, h] = (255, 255, 255, 0)
        return img

    def __paste(self, bg_img, tl=None, mode='grid', lines=1, intv=None, num_boxes=6):
        """
        粘贴透明图
        输入:
            bg_img: 背景图片
            tl(tuple or list): 贴图区域左上角坐标
            method: 贴图模式
            lines: 贴图行数（'grid'模式参数）
            intv(tuple or list): 每行贴图间隔系数（以背景图的宽度作为基准，'grid'模式参数）
            num_boxes: 贴图个数（'random'模式参数）
        输出:
            result: 生成图片
            rows(list): 标签信息，'grid'模式下：shape=(真实贴图行数, 真实贴图列数, 4, 2)，此时不一定是张量形式；
            'random'模式下：shape=(1, 贴图个数, 4, 2)，此时是张量形式
        """
        bg = Image.fromarray(cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB))  # 图片opencv格式转pillow格式
        # 默认原指定区域左上角作为贴图起始位置
        if tl is None:
            tl = (self.roi[0], self.roi[1])
        xy = list(tl)
        rows = []
        if mode == 'grid':
            # 默认每行间隔系数相同
            if intv is None:
                param = 0.1
                intv = np.array([param])
                step = np.repeat(intv, lines, axis=0) * bg.size[0]  # 贴图间隔
            else:
                assert len(intv) == lines, '贴图间隔系数应当与行数对应！'
                step = np.asarray(list(intv)) * bg.size[0]
            flag = False  # 判断贴图是否已超出原指定区域
            for i in range(lines):
                row = []
                # 每行贴图起始位置（可调节）
                xy[0] += random.randint(0, int((self.roi[0] + self.roi[2] - xy[0]) / 8))
                xy[1] += random.randint(0, int((self.roi[1] + self.roi[3] - xy[1]) / 8))
                # 记录每行贴图分割线位置
                border = 0
                # 每行等间隔贴透明图（模板图间隔可能不严格相等）
                while True:
                    tpl = cv2.imread(random.choice(self.tpl_path))
                    rot_fg, rbox = self.rotate_image(tpl, random.uniform(0., 360.))  # 随机将模板旋转0至360度
                    # 判断贴图是否已超出原指定区域
                    if xy[0] + rot_fg.shape[1] > (self.roi[0] + self.roi[2] - 10):
                        xy[0] = tl[0]
                        xy[1] = border
                        break
                    if xy[1] + rot_fg.shape[0] > (self.roi[1] + self.roi[3] - 10):
                        flag = True
                        break
                    # 透明贴图
                    fg = Image.fromarray(cv2.cvtColor(rot_fg, cv2.COLOR_BGR2RGB))
                    fg = self.to_transparent(fg)
                    bg.paste(fg, tuple(xy), mask=fg.split()[3])
                    # 记录位置信息
                    rbox[:, 0] += xy[0]
                    rbox[:, 1] += xy[1]
                    row.append(list(rbox))
                    # 坐标矫正
                    border = border if border > xy[1] + fg.height \
                        else xy[1] + fg.height
                    xy[0] += fg.width + int(step[i])
                if len(row) != 0:
                    rows.append(row)
                if flag:
                    break
        elif mode == 'random':
            row = []
            # 记录贴图失败的次数
            times = 0
            # 记录已贴图区域
            first = True
            old = []
            while True:
                # 若贴图失败次数已达上限，强制退出
                if times > 100:
                    print("随机模式下，已尝试去贴图，但有效贴图次数仍未能达到预期！")
                    break
                new = []
                # 矫正坐标（可调节）
                xy[0] = random.randint(tl[0], self.roi[0] + self.roi[2])
                xy[1] = random.randint(tl[1], self.roi[1] + self.roi[3])
                # 原指定区域内完全随机贴图
                tpl = cv2.imread(random.choice(self.tpl_path))
                rot_fg, rbox = self.rotate_image(tpl, random.uniform(0., 360.))  # 随机将模板旋转0至360度
                # 判断贴图是否已超出原指定区域
                if xy[0] + rot_fg.shape[1] > (self.roi[0] + self.roi[2] - 5) or \
                        xy[1] + rot_fg.shape[0] > (self.roi[1] + self.roi[3] - 5):
                    times += 1
                    continue
                if not first:
                    # 判断新贴图与已贴图区域的重叠程度（矩形区域）
                    new.append([xy[0], xy[1], xy[0] + rot_fg.shape[1], xy[1] + rot_fg.shape[0]])
                    iou = self.compute_intersection(np.asarray(new), np.asarray(old))  # (1, num_boxes)
                    if iou[0][np.argmax(iou[0, :])] > 0.01:  # 阈值
                        times += 1
                        continue
                # 透明贴图
                fg = Image.fromarray(cv2.cvtColor(rot_fg, cv2.COLOR_BGR2RGB))
                fg = self.to_transparent(fg)
                bg.paste(fg, tuple(xy), mask=fg.split()[3])
                # 记录位置信息
                rbox[:, 0] += xy[0]
                rbox[:, 1] += xy[1]
                row.append(list(rbox))
                # 贴图成功次数
                num_boxes -= 1
                if num_boxes == 0:
                    break
                # 更新已贴图区域
                if first:
                    new.append([xy[0], xy[1], xy[0] + rot_fg.shape[1], xy[1] + rot_fg.shape[0]])
                    first = False
                old.append(new[0])
            if len(row) != 0:
                rows.append(row)
        else:
            print("贴图失败！原因：模式不支持！")
            return bg_img, rows
        result = cv2.cvtColor(np.asarray(bg), cv2.COLOR_RGB2BGR)  # 图片pillow格式转opencv格式
        return result, rows

    def __save_data(self, idx, result, rows):
        """
        保存生成图片和标签txt
        输入：
            idx: 图片名索引
            result: 合成图
            rows: 标签信息
        """
        prefix = os.path.basename(os.path.splitext(self.bg_path)[0])
        suffix = os.path.splitext(self.bg_path)[1]
        img_file = prefix + "_{:0>4d}".format(idx) + suffix
        lbl_file = img_file.replace('.jpg', '.txt')
        # 图片路径
        img_path = os.path.join(self.sv_img_dir, img_file)
        # 标签路径
        lbl_path = os.path.join(self.sv_lbl_dir, lbl_file)
        # 存储合成图
        cv2.imwrite(img_path, result)
        # 读取类别
        category = os.path.basename(self.tpl_dir)
        assert category != '', "读取模板类别时出错！"
        # 存储标签
        with open(lbl_path, 'w') as lbl:
            for row in rows:
                for col in row:
                    lbl.write(' '.join([str(col[0][0]), str(col[0][1]), str(col[1][0]), str(col[1][1]),
                                        str(col[2][0]), str(col[2][1]), str(col[3][0]), str(col[3][1]),
                                        category]) + '\n')

    def generate(self, count, mode='random', lines=1, intv=None, num_boxes=6):
        """
        制作合成图
        输入：
            count: 每张背景图对应count张合成图
            mode: 贴图模式（有'grid'和'random'两种，默认为'grid'）
            lines: 'grid'模式的期望贴图行数
            intv: 'grid'模式下，每行贴图的间隔系数（以对应背景图的宽作为基准，默认是0.1）
            num_boxes: 'random'模式的期望贴图个数
        """
        for bg_file in os.listdir(self.bg_dir):
            self.bg_path = os.path.join(self.bg_dir, bg_file)
            bg_img = cv2.imread(self.bg_path, 1)
            # 指定贴图区域
            print("请选择贴图区域！")
            self.roi = cv2.selectROI('roi', bg_img, False)
            for i in range(count):
                # 指定区域内贴图起始位置（可调节）
                init_x = random.randint(self.roi[0], self.roi[0] + int(self.roi[2] / 6))
                init_y = random.randint(self.roi[1], self.roi[1] + int(self.roi[3] / 6))
                image, label = self.__paste(bg_img, (init_x, init_y),
                                            mode=mode, lines=lines, intv=intv, num_boxes=num_boxes)
                # image, label = self.__paste_rgba(bg_img, mode='random', num_boxes=4)
                self.__save_data(i + 1, image, label)
                print("贴图完毕！")


# 测试
if __name__ == '__main__':
    # data_maker = DataMaker("backgrounds",
    #                        "templates")
    # data_maker.generate(5, mode='grid', lines=2)
    img_path = "images/.."
    label_path = "labels/.."
    show_image(img_path, label_path)
