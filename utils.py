import json
import math
import os

import cv2
import numpy as np


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_json(filename):
    with open(filename, 'r', encoding='utf8') as f:
        return json.load(f)


def json_dumps(data, filename):
    dumps = json.dumps(data, ensure_ascii=False)  # 注意此处默认中文使用ascii编码，需要手动修改
    with open(filename, 'w', encoding='utf-8') as f:  # 以utf-8写入文件
        f.write(dumps + '\n')


def bbox2points(bbox):
    points = []
    for i in range(0, len(bbox), 2):
        points.append([bbox[i], bbox[i+1]])
    points = np.array(points, dtype=np.int64)
    return points


def find_min_area_rect(bbox):
    points = bbox2points(bbox)

    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    new_box = []
    for point in points:
        d0 = calculate_distance(point, box[0])
        d1 = calculate_distance(point, box[1])
        d2 = calculate_distance(point, box[2])
        d3 = calculate_distance(point, box[3])

        idx = np.argmin([d0, d1, d2, d3])
        new_box.append(box[idx])

    return new_box


def calculate_distance(p1, p2):
    d_x = p1[0] - p2[0]
    d_y = p1[1] - p2[1]
    return math.sqrt(d_x**2 + d_y**2)


def perspective(img, points):
    pts1 = np.float32(points)

    w = calculate_distance(pts1[0], pts1[1])
    h = calculate_distance(pts1[1], pts1[2])

    w = np.int(w)
    h = np.int(h)

    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (w, h))

    return dst


def paint_boundingbox(img, bbox, color=(0, 0, 255)):
    new_img = img.copy()

    points = bbox2points(bbox)
    cv2.drawContours(new_img, [points], 0, color=color,
                     thickness=1, lineType=cv2.LINE_AA)

    cv2.imshow('bbox', new_img)
    cv2.waitKey()


def paint_min_area_rect(img, bbox, color=(0, 0, 255)):
    box = find_min_area_rect(bbox)
    new_img = img.copy()
    cv2.drawContours(new_img, [box], 0, color=color,
                     thickness=1, lineType=cv2.LINE_AA)
    cv2.imshow('minAreaRect', new_img)
    cv2.waitKey()
