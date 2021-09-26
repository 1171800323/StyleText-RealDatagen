import argparse
import os
import random

import cv2
import numpy as np

from utils import makedirs
import skeletonization


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


def datagen(cfg):
    mask_path = os.path.join(cfg.textseg_extract_word_path, cfg.word_mask_path)

    surf_i_s_path = os.path.join(
        cfg.textseg_extract_word_path, cfg.word_fg_path)
    surf_i_s_list = os.listdir(surf_i_s_path)

    bg_filepath = cfg.bg_filepath
    bg_list = open(bg_filepath, 'r', encoding='utf8').readlines()
    bg_list = [img_path.strip() for img_path in bg_list]

    while True:
        bg = cv2.imread(random.choice(bg_list))

        surf_i_s_name = random.choice(surf_i_s_list)
        surf_i_s = cv2.imread(os.path.join(surf_i_s_path, surf_i_s_name))
        surf_h, surf_w = surf_i_s.shape[:2]
        print(surf_h, surf_w)

        if surf_h < cfg.min_h:
            continue

        bg_h, bg_w = bg.shape[:2]
        if bg_w < surf_w or bg_h < surf_h:
            continue
        x = np.random.randint(0, bg_w - surf_w + 1)
        y = np.random.randint(0, bg_h - surf_h + 1)
        t_b = bg[y:y + surf_h, x:x + surf_w, :]

        mask = cv2.imread(os.path.join(mask_path, surf_i_s_name))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        mask_bg = cv2.bitwise_and(t_b, t_b, mask=mask_inv)
        i_s = cv2.add(mask_bg, surf_i_s)

        sk = skeletonization.skeletonization(mask)
        cv2.imshow('is', i_s)
        cv2.imshow('mask', mask)
        cv2.imshow('sk', sk)
        cv2.waitKey()

        break


def save_data(cfg):
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


if __name__ == '__main__':
    config = get_parameters()
    print(config)
    datagen(config)
