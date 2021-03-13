import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def read_dicom(path, voi_lut=True, fix_monochrome=True):
    dicom_file = pydicom.read_file(path)

    if voi_lut:
        data = apply_voi_lut(dicom_file)
    else:
        data = dicom_file.pixel_array
        
    if fix_monochrome:
        data = data - np.amax(data)

    data = (data - data.min()) / (data.max() - data.min())

    return data


def resize_img(img_path, save_dir, size, keep_ratio=True, resample=Image.LANCZOS):
    img_name = os.path.split(img_path)[-1]
    img = Image.open(img_path)

    if keep_ratio:
        img.thumbnail((size, size), resample)
    else:
        img = img.resize((size, size), resample)

    img.save(os.path.join(save_dir, img_name))

    return img


def load_images():
    img_id, dim0, dim1 = [], [], []
    
    for split in ['train', 'test']:
        load_dir = f'{split}/'
        save_dir = f'/kaggle/images/{split}/'

        os.makedirs(save_dir, exist_ok=True)

        for file in tqdm(os.listdir(load_dir)):
            xray = read_dicom(load_dir + file)
            img = resize(xray, 512)
            img.save(save_dir + file.replace('dicom', 'png'))

            if split == 'train':
                img_id.append(file.replace('.dicom', ''))
                dim0.append(xray.shape[0])
                dim1.append(xray.shape[1])

    return img_id, (dim0, dim1)


if __name__ == '__main__':
    SAVE_DIR = f'nih1024'

    df = pd.read_csv('BBox_List_2017.csv')
    img_list = [os.path.join('', img) for img in df['Image Index'].unique()]
    os.makedirs(SAVE_DIR, exist_ok=True)

    for path in tqdm(img_list):
       resize_img(path, SAVE_DIR, size=1024)
