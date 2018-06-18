import time
import glob
import os
import cv2
import numpy as np
import pandas as pd
import h5py

BUCKET_NAME = "baidu-segmentation-dataset"
__all__ = ['generate_data']

def generate_data(input_size, image_dir, profile_dir, h5_path, df_path):
    print("start to generate dataset")
    df = pd.read_csv(df_path)

    # 데이터셋은 영상에 한명만 있는 경우만 포함함
    filename_series = df[df.num_person == 1].filename
    # image를 담을 데이터 셋
    # dataset-shape : the number of data, height, width, channel
    dataset = np.zeros((len(filename_series), *input_size, 4),dtype=np.uint8)

    start_time = time.time()
    for idx, filename in enumerate(filename_series):
        # image와 profile 가져오기
        image_path = os.path.join(image_dir,filename)
        profile_path = os.path.join(profile_dir,filename)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        profile = cv2.imread(profile_path, 0)
        # profile을 Color channel 중 4번째에 추가
        img_and_profile = np.concatenate([img,np.expand_dims(profile,axis=-1)],axis=2)
        # Padding을 추가시켜 정방행렬화
        h, w, _ = img_and_profile.shape
        if h - w > 0:
            diff = h - w
            pad1, pad2 = diff // 2 , diff - diff//2
            pad_input = np.pad(img_and_profile,((0,0),(pad1,pad2),(0,0)),'constant',constant_values=255)
        elif h - w < 0:
            diff = w - h
            pad1, pad2 = diff // 2 , diff - diff//2
            pad_input = np.pad(img_and_profile,((pad1,pad2),(0,0),(0,0)),'constant',constant_values=255)
        else:
            pad_input = img_and_profile
        # Resize함
        resized = cv2.resize(pad_input, input_size)
        # dataset에 담음
        dataset[idx] = resized
        if idx % 100 == 0:
            print(idx,"th completed --- time : {}".format(time.time()-start_time))

    # dataset 저장하기
    with h5py.File(h5_path) as file:
        file.create_dataset("{}x{}".format(*input_size),
                            data=dataset,dtype=np.uint8)
    print("save dataset in {}".format(h5_path))

    # dataset 저장하기
    file = h5py.File(h5_path)
    file.create_dataset('384x384',data=dataset,dtype=np.uint8)
    file.close()

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
    start_time = time.time()
    for idx, key in enumerate(keys):
        filename = os.path.split(key)[1]
        if "images" in key:
            file_path = os.path.join(image_dir, filename)
        elif "profiles" in key:
            file_path = os.path.join(label_dir, filename)
        else:
            continue

        download(s3, BUCKET_NAME, key, file_path)
        if idx % 100 == 0:
            print("{} download is completed -- {:.2f}".format(idx, time.time()-start_time))
            start_time = time.time()
    return image_dir, label_dir
