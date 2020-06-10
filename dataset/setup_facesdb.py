from PIL import Image, ImageOps, ImageEnhance
import pandas as pd
import numpy as np
import random
import os

############################################################
#  このスクリプトはデータセットと同じディレクトリに置いて使ってください．
############################################################


# 画像の中央を切り抜く
def crop_center(file_path, crop_width, crop_height):
    # https://note.nkmk.me/python-pillow-image-crop-trimming/
    image = Image.open(file_path)
    img_width, img_height = image.size
    random_offset_width = 20
    horizontal_offset = random.randint(-random_offset_width, random_offset_width) + 45
    vertical_offset = random.randint(-random_offset_width, random_offset_width)
    is_random = True
    if is_random:
        trimmed_image = image.crop(((img_width - crop_width) // 2 + vertical_offset,
                            (img_height - crop_height) // 2 + horizontal_offset,
                            (img_width + crop_width) // 2 + vertical_offset,
                            (img_height + crop_height) // 2 + horizontal_offset))
    else:
        trimmed_image = image.crop(((img_width - crop_width) // 2,
                       (img_height - crop_height) // 2 + 45,
                       (img_width + crop_width) // 2,
                       (img_height + crop_height) // 2 + 45))
    
    # ランダムに画像を左右反転
    if random.random() > 0.5:
        trimmed_image = ImageOps.mirror(trimmed_image)

    # ランダムに明るさを変更
    # brightness_altered_image = ImageEnhance.Brightness(trimmed_image)
    # brightness_altered_image = brightness_altered_image.enhance(random.uniform(0.7, 1.3))
    # return brightness_altered_image
    return trimmed_image


def create_dataset(is_split):
    working_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(working_dir, "facesdb")  # カラー画像のデータセット
    # train用のディレクトリとtest用のディレクトリを入れるディレクトリ
    out_dir = os.path.join(working_dir, "facesdb_prepared")
    train_data_dir = os.path.join(out_dir, "train_data")  # train用の画像を入れるディレクトリ
    test_data_dir = os.path.join(out_dir, "test_data")  # test用の画像を入れるディレクトリ

    os.makedirs(train_data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    train_dict = {"Class Label": [], "File Path": []}
    test_dict = {"Class Label": [], "File Path": []}
    image_length = 240
    copy_num = 1
    for emotion_id in range(7):
        (train_data_index, test_data_index) = randomly_split_data(emotion_id, is_split)
        for person_id_train in train_data_index:
            input_file_train = "s{0:03d}-{1:02d}_img.bmp".format(person_id_train, emotion_id)
            for i in range(copy_num):
                file_path = os.path.join(root_dir, "s{0:03d}".format(person_id_train), "bmp", input_file_train)
                # 両側の黒い部分を切り取る
                croped_image = crop_center(file_path, image_length, image_length)
                output_file_train = "s{0:03d}-{1:02d}_{2}img.bmp".format(person_id_train, emotion_id, i)
                croped_image.save(os.path.join(train_data_dir, output_file_train))
                train_dict["Class Label"].append(put_label(emotion_id, is_split))
                train_dict["File Path"].append(os.path.join("train_data/", output_file_train))

        for person_id_test in test_data_index:
            file_name_test = "s{0:03d}-{1:02d}_img.bmp".format(person_id_test, emotion_id)
            file_path = os.path.join(root_dir, "s{0:03d}".format(person_id_test), "bmp", file_name_test)
            # 両側の黒い部分を切り取る
            croped_image = crop_center(file_path, image_length, image_length)
            croped_image.save(os.path.join(test_data_dir, file_name_test))
            test_dict["Class Label"].append(put_label(emotion_id, is_split))
            test_dict["File Path"].append(os.path.join("test_data/", file_name_test))

    df = pd.DataFrame(data=train_dict)
    df.to_csv(os.path.join(out_dir, "train_list.csv"), index=False)
    df = pd.DataFrame(data=test_dict)
    df.to_csv(os.path.join(out_dir, "test_list.csv"), index=False)

def randomly_split_data(emotion_id, is_split):
    person_labels = []
    for i in range(1, 39):
        if i != 20 and i != 22:
            person_labels.append(i)
    random.shuffle(person_labels)
    if is_split:
        if emotion_id == 2 or emotion_id == 4 or emotion_id == 5 or emotion_id == 6:
            return (person_labels[0:6], person_labels[6:9])
        else:
            return (person_labels[0:24], person_labels[24:])
    else:
        return (person_labels[0:24], person_labels[24:])

def put_label(n, is_split):
    assert(0 <= n < 7)
    if is_split:
        if n == 4 or n == 5 or n == 6:
            return 2
        else:
            return n
    else:
        return n

if __name__ == "__main__":
    is_split = True
    create_dataset(is_split)
    print("Successful.")
