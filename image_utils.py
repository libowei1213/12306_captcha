# coding=utf-8
from PIL import Image
import numpy as np
import os
import time
import requests
import uuid

session = requests.Session()
requests.packages.urllib3.disable_warnings()


def read_data(image_dir, image_shape, label_path="labels.txt"):
    """
    读取图片
    :param image_dir:
    :param image_shape:
    :param label_path:
    :return:
    """
    label_dict = {}
    if os.path.exists(label_path):
        with open(label_path, encoding="utf-8") as file:
            for line in file:
                class_name, id = line.strip().split()
                label_dict[class_name] = int(id)
    else:
        with open(label_path, "w", encoding="utf-8") as file:
            for id, class_name in enumerate(os.listdir(image_dir)):
                file.write("%s %s\n" % (class_name, id))
                label_dict[class_name] = id

    images = []
    labels = []

    file_label_dict = {}
    for dir_name in os.listdir(image_dir):
        for filename in os.listdir(os.path.join(image_dir, dir_name)):
            full_path = os.path.join(image_dir, dir_name, filename)
            label = label_dict[dir_name]
            file_label_dict[full_path] = label
    keys = list(file_label_dict.keys())
    np.random.shuffle(keys)

    for path in keys:
        image = Image.open(path)
        if image.size != image_shape[:2]:
            image = image.resize(image_shape[:2])
        image = np.asarray(image)
        images.append(image)
        labels.append(file_label_dict[path])

    return np.array(images), np.array(labels)


def download_captcha(retry=False):
    """
    下载验证码图片
    :param retry:
    :return:
    """
    url = "https://kyfw.12306.cn/passport/captcha/captcha-image"
    r = session.get(url, verify=False)
    if retry:
        url2 = "https://kyfw.12306.cn/passport/captcha/captcha-check?answer=129%2C122%2C175%2C132&login_site=E&rand=sjrand"
        time.sleep(3)
        session.get(url2, verify=False)
        time.sleep(3)
        r = session.get(url, verify=False)
    filename = "temp/%s.png" % uuid.uuid4()
    with open(filename, "wb") as file:
        file.write(r.content)
    return filename


def submit_captcha(ids):
    """
    提交验证码
    :param ids:
    :return:
    """
    indices = {0: "48,70",
               1: "100,70",
               2: "180,70",
               3: "250,70",
               4: "48,150",
               5: "100,150",
               6: "180,150",
               7: "250,150"}

    list = [indices[id] for id in ids]
    answer = ",".join(list)
    post = {
        "answer": answer,
        "login_site": "E",
        "rand": "sjrand"
    }

    url = "https://kyfw.12306.cn/passport/captcha/captcha-check"
    s = session.post(url, data=post, verify=False)
    return s.text


def process_raw_images(raw_image, image_shape):
    flag = judge_image_background(raw_image)
    text_part = split_image_text(raw_image, image_shape, flag)
    image_part = cut_images(raw_image, image_shape)
    return text_part, image_part


def save(image, dir, label, image_shape=(64, 64)):
    """
    保存图片
    :param image:
    :param dir:  保存目录
    :param label: 子文件夹
    :param image_shape:
    :return:
    """
    if not os.path.exists(dir):
        os.mkdir(dir)

    path = os.path.join(dir, label)
    if not os.path.exists(path):
        os.mkdir(path)

    filename = "%s.png" % uuid.uuid4()
    image = image.resize(image_shape)
    image.save(os.path.join(path, filename))


def cut_images(raw_image, image_shape):
    """
    切分出8个验证码
    :param raw_image:
    :return:
    """
    crop_area = [(5, 41, 71, 107), (77, 41, 143, 107), (149, 41, 215, 107), (221, 41, 287, 107),
                 (5, 113, 71, 179), (77, 113, 143, 179), (149, 113, 215, 179), (221, 113, 287, 179)]
    if isinstance(raw_image, str):
        raw_image = Image.open(raw_image)
    return [raw_image.crop(region).resize(image_shape) for region in crop_area]


def judge_image_background(raw_image):
    """
    判断验证码文字区域的词个数
    :param raw_image: Image对象或图像路径
    :return: 1 或 2
    """
    if isinstance(raw_image, str):
        raw_image = Image.open(raw_image)

    # 裁切出验证码文字区域 高28 宽112
    image = raw_image.crop((118, 0, 230, 28))
    image = image.convert("P")
    image_array = np.asarray(image)

    # 取最后4行
    image_array = image_array[24:28]

    if np.mean(image_array) > 200:
        return 1
    else:
        return 2


def split_image_text(raw_image, image_shape, mode=1):
    """
    裁切出验证码文字部分
    :param raw_image: Image对象或图像路径
    :param mode: 图中有几组验证码文字
    :return:
    """
    if isinstance(raw_image, str):
        raw_image = Image.open(raw_image)
    # 裁切出验证码文字区域 高28 宽112
    image = raw_image.crop((118, 0, 230, 28))

    resize_list = []
    if mode == 1:
        # 图中只有一组验证码
        image_array = np.asarray(image)
        image_array = image_array[6:22]
        image_array = np.mean(image_array, axis=2)
        image_array = np.mean(image_array, axis=0)
        image_array = np.reshape(image_array, [-1])

        indices = np.where(image_array < 240)
        resize_list.append((indices[0][0], indices[0][-1]))

    if mode == 2:
        # 图中只有两组验证码
        image_p = image.convert("P")
        image_array = np.asarray(image_p)
        image_array = image_array[6:22]
        image_array = np.mean(image_array, axis=0)
        avg_image = np.reshape(image_array, [-1])
        indices = np.where(avg_image < 190)
        start = indices[0][0] - 1
        end = indices[0][0] - 1
        for i in indices[0]:
            if i == end + 1:
                end = i
            else:
                if end - start > 10:
                    resize_list.append([start + 1, end])
                start = i
                end = i
        if end - start > 10:
            resize_list.append([start + 1, end])

    return [image.crop((x1, 0, x2, 28)).resize(image_shape) for x1, x2 in resize_list]


if __name__ == '__main__':
    pass
