import argparse
import multiprocessing
import os
import random

import cv2
import numpy as np

import colorize
import skeletonization
from render_standard_text import make_standard_text
from utils import makedirs, read_json


class datagen():
    def __init__(self, cfg):
        self.mask_path = os.path.join(
            cfg.textseg_extract_word_path, cfg.word_mask_path)

        self.font_path = cfg.font_path

        word_json_path = os.path.join(
            cfg.textseg_extract_word_path, cfg.word_json_path)
        word_json_name = os.listdir(word_json_path)[0]
        self.word_json = read_json(
            os.path.join(word_json_path, word_json_name))

        self.surf_i_s_path = os.path.join(
            cfg.textseg_extract_word_path, cfg.word_fg_path)
        self.surf_i_s_list = os.listdir(self.surf_i_s_path)

        bg_filepath = cfg.bg_filepath
        bg_list = open(bg_filepath, 'r', encoding='utf8').readlines()
        self.bg_list = [img_path.strip() for img_path in bg_list]

        self.min_h = cfg.min_h

    def gen_styletext_data(self):

        while True:
            bg = cv2.imread(random.choice(self.bg_list))

            surf_i_s_name = random.choice(self.surf_i_s_list)
            surf_i_s = cv2.imread(os.path.join(
                self.surf_i_s_path, surf_i_s_name))
            surf_h, surf_w = surf_i_s.shape[:2]
            # print(surf_h, surf_w)

            if surf_h < self.min_h:
                continue

            bg_h, bg_w = bg.shape[:2]
            if bg_w < surf_w or bg_h < surf_h:
                continue
            x = np.random.randint(0, bg_w - surf_w + 1)
            y = np.random.randint(0, bg_h - surf_h + 1)
            t_b = bg[y:y + surf_h, x:x + surf_w, :]

            mask = cv2.imread(os.path.join(self.mask_path, surf_i_s_name))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

            # 简单叠加
            mask_inv = cv2.bitwise_not(mask)
            mask_bg = cv2.bitwise_and(t_b, t_b, mask=mask_inv)
            i_s = cv2.add(mask_bg, surf_i_s)

            # 泊松融合，效果不好
            # center = (surf_w // 2, surf_h // 2)
            # seamlessClone_mask = 255 * np.ones(surf_i_s.shape, surf_i_s.dtype)
            # i_s = cv2.seamlessClone(
            #     surf_i_s, t_b, seamlessClone_mask, center, cv2.MIXED_CLONE)

            t_sk = skeletonization.skeletonization(mask)

            t_t = colorize.color(mask, surf_i_s, (127, 127, 127))

            text = self.word_json[surf_i_s_name]
            i_t = make_standard_text(self.font_path, text, (surf_h, surf_w))

            t_f = i_s.copy()
            break

        return i_t, i_s, t_sk, t_t, t_b, t_f, mask


def enqueue_data(datagen, queue, capacity):
    np.random.seed()

    while True:
        try:
            data = datagen.gen_styletext_data()
        except Exception as e:
            pass
        else:
            if queue.qsize() < capacity:
                queue.put(data)


class multiprocess_datagen():
    def __init__(self, process_num, data_capacity, datagen):
        self.datagen = datagen
        self.process_num = process_num
        self.data_capacity = data_capacity

    def multiprocess_runningqueue(self):

        manager = multiprocessing.Manager()

        # 对于所有进程维护一个队列
        self.queue = manager.Queue()

        # 使用进程池批量创建子进程
        self.pool = multiprocessing.Pool(processes=self.process_num)
        self.processes = []
        for _ in range(self.process_num):
            p = self.pool.apply_async(
                enqueue_data, args=(self.datagen, self.queue, self.data_capacity))
            self.processes.append(p)
        self.pool.close()

    def dequeue_data(self):

        while self.queue.empty():
            pass
        data = self.queue.get()  # 从队列取出一个项目
        return data

    def get_queue_size(self):
        return self.queue.qsize()

    def terminate_pool(self):
        self.pool.terminate()


def get_parameters():
    parser = argparse.ArgumentParser()
    # 源数据地址
    parser.add_argument('--textseg_extract_word_path', type=str,
                        default='D:/Code/datasets/TextSeg_Extract_Word')
    parser.add_argument('--word_image_path', type=str, default='image')
    parser.add_argument('--word_json_path', type=str, default='json')
    parser.add_argument('--word_mask_path', type=str, default='mask')
    parser.add_argument('--word_fg_path', type=str, default='fg')

    parser.add_argument('--bg_filepath', type=str,
                        default='D:/Code/DeepLearning/datasets/srnet_bg/labels.txt')
    parser.add_argument('--font_path', type=str,
                        default='fonts/zh_standard.ttc')

    parser.add_argument('--process_num', type=int, default=4)
    parser.add_argument('--data_capacity', type=int, default=256)
    parser.add_argument('--sample_num', type=int, default=10)

    parser.add_argument('--min_h', type=int, default=20)
    # 处理后的数据地址
    parser.add_argument('--data_dir', type=str,
                        default='D:/Code/datasets/StyleText_RealData')
    parser.add_argument('--i_s_dir', type=str, default='i_s')
    parser.add_argument('--i_t_dir', type=str, default='i_t')
    parser.add_argument('--t_t_dir', type=str, default='t_t')
    parser.add_argument('--t_b_dir', type=str, default='t_b')
    parser.add_argument('--t_f_dir', type=str, default='t_f')
    parser.add_argument('--mask_t_dir', type=str, default='mask_t')
    parser.add_argument('--t_sk_dir', type=str, default='t_sk')

    return parser.parse_args()


def main():
    cfg = get_parameters()
    print(cfg)

    i_t_dir = os.path.join(cfg.data_dir, cfg.i_t_dir)
    i_s_dir = os.path.join(cfg.data_dir, cfg.i_s_dir)
    t_sk_dir = os.path.join(cfg.data_dir, cfg.t_sk_dir)
    t_t_dir = os.path.join(cfg.data_dir, cfg.t_t_dir)
    t_b_dir = os.path.join(cfg.data_dir, cfg.t_b_dir)
    t_f_dir = os.path.join(cfg.data_dir, cfg.t_f_dir)
    mask_t_dir = os.path.join(cfg.data_dir, cfg.mask_t_dir)

    makedirs(i_t_dir)
    makedirs(i_s_dir)
    makedirs(t_sk_dir)
    makedirs(t_t_dir)
    makedirs(t_b_dir)
    makedirs(t_f_dir)
    makedirs(mask_t_dir)

    mp_gen = multiprocess_datagen(
        process_num=cfg.process_num, data_capacity=cfg.data_capacity, datagen=datagen(cfg))

    mp_gen.multiprocess_runningqueue()
    digit_num = len(str(cfg.sample_num)) - 1
    for idx in range(cfg.sample_num):
        print(
            "Generating step {:>6d} / {:>6d}".format(idx + 1, cfg.sample_num))
        i_t, i_s, t_sk, t_t, t_b, t_f, mask_t = mp_gen.dequeue_data()
        i_t_path = os.path.join(i_t_dir, str(idx).zfill(digit_num) + '.png')
        i_s_path = os.path.join(i_s_dir, str(idx).zfill(digit_num) + '.png')
        t_sk_path = os.path.join(t_sk_dir, str(idx).zfill(digit_num) + '.png')
        t_t_path = os.path.join(t_t_dir, str(idx).zfill(digit_num) + '.png')
        t_b_path = os.path.join(t_b_dir, str(idx).zfill(digit_num) + '.png')
        t_f_path = os.path.join(t_f_dir, str(idx).zfill(digit_num) + '.png')
        mask_t_path = os.path.join(
            cfg.data_dir, cfg.mask_t_dir, str(idx).zfill(digit_num) + '.png')
        cv2.imwrite(i_t_path, i_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(i_s_path, i_s, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(t_sk_path, t_sk, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(t_t_path, t_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(t_b_path, t_b, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(t_f_path, t_f, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(mask_t_path, mask_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    mp_gen.terminate_pool()


if __name__ == '__main__':
    main()
