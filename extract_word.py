import argparse
import os

import cv2
import numpy as np

from utils import (find_min_area_rect, json_dumps, makedirs, perspective,
                   read_json)


def produce_word_fg(image, mask):
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    idx = (mask_gray != 0)

    h, w, c = image.shape
    new_img = np.zeros((h, w, c))

    for i in range(c):
        new_img[:, :, i] = image[:, :, i] * idx
    new_img = np.uint8(new_img)

    return new_img


def main(args):
    # 源数据地址
    image_path = os.path.join(args.textseg_path, args.image_path)
    annotation_path = os.path.join(args.textseg_path, args.annotation_path)
    semantic_label_path = os.path.join(
        args.textseg_path, args.semantic_label_path)

    # 处理后的数据地址
    word_image_path = os.path.join(
        args.textseg_extract_word_path, args.word_image_path)
    word_json_path = os.path.join(
        args.textseg_extract_word_path, args.word_json_path)
    word_mask_path = os.path.join(
        args.textseg_extract_word_path, args.word_mask_path)
    word_fg_path = os.path.join(
        args.textseg_extract_word_path, args.word_fg_path)

    makedirs(word_image_path)
    makedirs(word_json_path)
    makedirs(word_mask_path)
    makedirs(word_fg_path)

    word_json = {}

    image_name_list = os.listdir(image_path)

    for image_name in image_name_list:
        image = cv2.imread(os.path.join(image_path, image_name))

        image_name_idx = image_name[:-4]
        print(image_name_idx)

        semantic_label = cv2.imread(os.path.join(
            semantic_label_path, image_name_idx + '_maskfg.png'))

        annotation_json = read_json(os.path.join(
            annotation_path, image_name_idx + '_anno.json'))

        for key, value in annotation_json.items():
            text = value['text']
            bbox = value['bbox']

            min_area_rect_box = find_min_area_rect(bbox)

            word_image = perspective(image, min_area_rect_box)
            word_mask = perspective(semantic_label, min_area_rect_box)
            word_fg = produce_word_fg(word_image, word_mask)

            save_name = '{}_{}.png'.format(image_name_idx, key)
            word_json[save_name] = text

            cv2.imwrite(os.path.join(word_image_path, save_name), word_image)
            cv2.imwrite(os.path.join(word_mask_path, save_name), word_mask)
            cv2.imwrite(os.path.join(word_fg_path, save_name), word_fg)

    json_dumps(word_json, os.path.join(word_json_path, 'annotation.json'))


def get_parameters():
    parser = argparse.ArgumentParser()
    # 源数据地址
    parser.add_argument('--textseg_path', type=str,
                        default='D:/Code/datasets/TextSeg_Release')
    parser.add_argument('--image_path', type=str, default='image')
    parser.add_argument('--annotation_path', type=str, default='annotation')
    parser.add_argument('--semantic_label_path', type=str,
                        default='semantic_label')

    # 处理后的数据地址
    parser.add_argument('--textseg_extract_word_path', type=str,
                        default='D:/Code/datasets/TextSeg_Extract_Word')
    parser.add_argument('--word_image_path', type=str, default='image')
    parser.add_argument('--word_json_path', type=str, default='json')
    parser.add_argument('--word_mask_path', type=str, default='mask')
    parser.add_argument('--word_fg_path', type=str, default='fg')
    return parser.parse_args()


if __name__ == '__main__':
    config = get_parameters()
    print(config)
    # main(config)
