import time
import glob
import os
import cv2
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from data_utils import margin_to_square

BUCKET_NAME = "baidu-segmentation-dataset"
__all__ = ['generate_data']

def generate_data(img_dim, image_dir, profile_dir, h5_path, df_path,dataset_name):
    print("start to generate dataset")
    df = pd.read_csv(df_path)

    filename_series = df.filename
    # image를 담을 데이터 셋
    # dataset-shape : the number of data, height, width, channel
    dataset = np.zeros((len(filename_series), *img_dim[:2], 4),dtype=np.uint8)

    for idx, filename in tqdm(enumerate(filename_series)):
        # image와 profile 가져오기
        image_path = os.path.join(image_dir,filename)
        profile_path = os.path.join(profile_dir,filename)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        profile = cv2.imread(profile_path, 0)
        # flipping label
        profile = ~profile
        # profile을 Color channel 중 4번째에 추가
        img_and_profile = np.concatenate([img,np.expand_dims(profile,axis=-1)],axis=2)
        # 직사각형인 사진에 margin을 주어 정사각형으로 변형
        margined = margin_to_square(img_and_profile)
        # Resize함
        resized = cv2.resize(margined, img_dim[:2])
        # dataset에 담음
        dataset[idx] = resized

    # dataset 저장하기
    with h5py.File(h5_path) as file:
        if not dataset_name in file:
            file.create_dataset(dataset_name,
                            data=dataset,dtype=np.uint8)
        else:
            file[dataset_name][...] = dataset
    print("save dataset in {}".format(h5_path))

'''
    S3 Communication 관련 메소드
'''
def download(s3, bucket, obj, local_file_path):
    s3.download_file(bucket, obj,local_file_path)

def upload(s3, bucket, obj, local_file_path):
    s3.upload_file(local_file_path,bucket,obj)

def get_s3_keys(s3, bucket=BUCKET_NAME):
    # s3 버킷 내 key 집합을 구함
    keys = []
    res = s3.list_objects_v2(Bucket=bucket)
    while True:
        if not 'Contents' in res:
            break
        for obj in res['Contents']:
            keys.append(obj['Key'])

        last = res['Contents'][-1]['Key']
        res = s3.list_objects_v2(Bucket=BUCKET_NAME,StartAfter=last)
    return keys

def download_whole_dataset(s3, image_dir, label_dir):
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    keys = get_s3_keys(s3)
    for idx, key in tqdm(enumerate(keys)):
        filename = os.path.split(key)[1]
        if "images" in key:
            file_path = os.path.join(image_dir, filename)
        elif "profiles" in key:
            file_path = os.path.join(label_dir, filename)
        else:
            continue
        download(s3, BUCKET_NAME, key, file_path)

    return image_dir, label_dir
