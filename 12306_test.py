from model.densenet import DenseNet
import image_utils
import numpy as np
import time
import shutil
import os

n_classes = 80
image_shape = (64, 64, 3)
text_model_weight = "saves/DenseNet-BC_k=12_d=40.weight"
image_model_weight = "saves/DenseNet-BC_k=24_d=40.weight"
save_path = "D:\IMAGE"
save_fail_path = "D:\IMAGE\FAIL"


def load_model():
    text_model = DenseNet(classes=n_classes, input_shape=image_shape, depth=40, growth_rate=12, bottleneck=True,
                          reduction=0.5, dropout_rate=0.0, weight_decay=1e-4)
    image_model = DenseNet(classes=n_classes, input_shape=image_shape, depth=40, growth_rate=24, bottleneck=True,
                           reduction=0.5, dropout_rate=0.0, weight_decay=1e-4)
    text_model.load_weights(text_model_weight)
    image_model.load_weights(image_model_weight)
    return text_model, image_model


def load_label_dict():
    # 读取类别名称
    label_dict = {}
    with open("labels.txt", encoding="utf-8") as file:
        for line in file:
            class_name, id = line.strip().split()
            label_dict[int(id)] = class_name
    return label_dict


def online_test(text_model, image_model, label_dict):
    """
    获取验证码图片、模型识别、提交
    :return:
    """

    # 下载验证码图片到本地
    image_path = image_utils.download_captcha()
    # 切割验证码为文字部分和图片部分
    raw_texts, raw_images = image_utils.process_raw_images(image_path, (image_shape[0], image_shape[1]))

    # 图像转换为np数组
    texts, images = np.array([np.asarray(image) for image in raw_texts]), np.array(
        [np.asarray(image) for image in raw_images])

    # 模型输出
    text_predict = text_model.predict(texts)
    image_predict = image_model.predict(images)

    # 预测结果
    text_result = np.argmax(text_predict, 1)
    image_result = np.argmax(image_predict, 1)

    # 概率
    text_prob = np.max(text_predict, 1)
    image_prob = np.max(image_predict, 1)

    # 类别名
    text_label = [label_dict[r] for r in text_result]
    image_label = [label_dict[r] for r in image_result]

    ids = set()
    for r1 in text_result:
        for id, r2 in enumerate(image_result):
            if r1 == r2:
                ids.add(id)

    result = image_utils.submit_captcha(ids)
    if "成功" in result:
        if save_path:
            # 保存图片
            for id in ids:
                image_utils.save(raw_images[id], os.path.join(save_path, "IMG"), label_dict[image_result[id]])
            for id, image in enumerate(raw_texts):
                image_utils.save(image, os.path.join(save_path, "TXT"), label_dict[text_result[id]])
            return True
    else:
        if save_fail_path:
            if not os.path.exists(save_fail_path):
                os.mkdir(save_fail_path)
            shutil.move(image_path, save_fail_path)
        return False


if __name__ == '__main__':
    text_model, image_model = load_model()
    label_dict = load_label_dict()

    test_result = {True: 0, False: 0}
    while True:
        try:
            test_result[online_test(text_model, image_model, label_dict)] += 1
            time.sleep(3)
            true_times, false_times = test_result[True], test_result[False]
            print("%d/%d 准确率:%.3f" % (true_times, true_times + false_times, true_times / (true_times + false_times)))
        except Exception:
            time.sleep(3)
