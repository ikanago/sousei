from PIL import Image
import pandas as pd
import numpy as np
import random
import os
import shutil

############################################################
#  このスクリプトはデータセットと同じディレクトリに置いて使ってください．
############################################################


def crop_center(file_path, crop_width, crop_height):
    # https://note.nkmk.me/python-pillow-image-crop-trimming/
    image = Image.open(file_path)
    img_width, img_height = image.size
    return image.crop(((img_width - crop_width) // 2,
                       (img_height - crop_height) // 2,
                       (img_width + crop_width) // 2,
                       (img_height + crop_height) // 2))


def create_dataset():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(working_dir, "facesdb")  # カラー画像のデータセット
    # train用のディレクトリとtest用のディレクトリを入れるディレクトリ
    out_dir = os.path.join(working_dir, "facesdb_prepared")
    train_data_dir = os.path.join(out_dir, "train_data")  # train用の画像を入れるディレクトリ
    test_data_dir = os.path.join(out_dir, "test_data")  # test用の画像を入れるディレクトリ

    dataset_size = 38 * 7  # 画像の総数
    test_data_ratio = 0.2  # データセット全体に対するテストデータの割合

    os.makedirs(train_data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    train_dict = {"Class Label": [], "File Path": []}
    test_dict = {"Class Label": [], "File Path": []}
    for person_dir in os.listdir(root_dir):
        if person_dir == 'test_data' or person_dir == 'train_data':
            continue
        current_dir = os.path.join(root_dir, person_dir, "bmp")
        for file_name in os.listdir(current_dir):
            # 両側の黒い部分を切り取る
            croped_image = crop_center(os.path.join(
                current_dir, file_name), 480, 480)
            # 乱数が`test_data_ratio`を超えた場合trainに使う
            if random.random() > test_data_ratio:
                croped_image.save(os.path.join(train_data_dir, file_name))
                new_file_name = os.path.join("train_data/", file_name)
                # ファイル名の6,7文字目が感情値に対応
                train_dict["Class Label"].append(int(file_name[5:7]))
                train_dict["File Path"].append(new_file_name)
            # 乱数が`test_data_ratio`以下の場合testに使う
            else:
                croped_image.save(os.path.join(test_data_dir, file_name))
                new_file_name = os.path.join("test_data/", file_name)
                test_dict["Class Label"].append(int(file_name[5:7]))
                test_dict["File Path"].append(new_file_name)
        shutil.rmtree(os.path.join(root_dir, person_dir))

    df = pd.DataFrame(data=train_dict)
    df.to_csv(os.path.join(out_dir, "train_list.csv"), index=False)
    df = pd.DataFrame(data=test_dict)
    df.to_csv(os.path.join(out_dir, "test_list.csv"), index=False)


if __name__ == "__main__":
    create_dataset()
    print("Successful.")
