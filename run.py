import os
import argparse
import random
from generator import ImageGenerator


def corpus_gen(dict_path, output_path, corpus_num, size_min=10, size_max=20):
    # dict_file = Path(dict_path)
    # output_file = Path(output_path)
    # if dict_file.is_file() and output_file.is_file():
    try:
        with open(dict_path, 'r') as dic:
            corpus_dic = [char for char in dic.read().splitlines() if len(char) > 0]
            # print(corpus_dic)
        with open(output_path, 'w') as out:
            for i in range(corpus_num):
                row_len = random.randrange(size_min, size_max)
                row = ''
                # corpus_dic.remove(row_first)
                for j in range(row_len):
                    row += random.choice(corpus_dic)
                row += '\n'
                out.write(row)
    except IOError:
        print("files io error!")
    else:
        print("done!")
    # else:
    #     print("io error!")


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gen_num",
        type=int,
        nargs="?",
        help="the num of the output data",
        default=1000
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        nargs="?",
        help="The output directory",
        default="output/",
    )
    parser.add_argument(
        "--font_dir",
        type=str,
        nargs="?",
        help="The font directory",
        default="fonts/test/"
    )
    parser.add_argument(
        "--img_size",
        type=tuple,
        nargs="?",
        help="size of generated image",
        default=(1280, 960)
    )
    # parser.add_argument(
    #     "--bg_color",
    #     nargs="?",
    #     help="background colour of generated image. Default is white",
    #     default=(255, 255, 255, 255)
    # )
    parser.add_argument(
        "--bg_path",
        nargs="?",
        help="path to the background img of generated image.",
        default="bgdata/0002.jpg"
    )
    parser.add_argument(
        "--fs_min",
        type=int,
        nargs="?",
        help="minimum font size of the genearted text",
        default=35
    )
    parser.add_argument(
        "--fs_max",
        type=int,
        nargs="?",
        help="maximum font size of the genearted text",
        default=75
    )
    # parser.add_argument(
    #     "--fs_size",
    #     type=int,
    #     nargs="?",
    #     help="font size of the genearted text",
    #     default=30
    # )
    parser.add_argument(
        "--text_corpus",
        type=str,
        nargs="?",
        help="path to the text corpus file",
        default="orgdata/test.txt"
    )
    parser.add_argument(
        "--date_corpus",
        type=str,
        nargs="?",
        help="path to the text corpus file",
        default="dicts/dates.txt"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # img_generator = ImageGenerator(args.font_dir, args.output_dir, args.fs_min, args.fs_max, args.text_corpus,
    #                                args.date_corpus, args.bg_path, args.img_size)
    # img_generator = ImageGenerator(args.font_dir, args.output_dir, args.fs_min, args.fs_max, args.text_corpus,
    # args.date_corpus, args.img_size, args.bg_color)
    dict_path = './dicts/test.txt'
    corpus_path = './orgdata/test.txt'
    divide = 10  # 每种字体样本数
    offset = 900  # 命名需要
    for i in range(args.gen_num // divide):
        corpus_gen(dict_path, corpus_path,  divide*10)
        img_generator = ImageGenerator(args.font_dir, args.output_dir, args.fs_min, args.fs_max, args.text_corpus,
                                       args.date_corpus, args.bg_path, args.img_size)
        img_generator.generateImages(divide, i*divide+1 + offset, 1, 3, draw_rectangle=False)


if __name__ == '__main__':
    main()
