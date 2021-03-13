import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import cv2
import numpy as np
import pandas as pd
import glob
import os.path as osp
from path import Path
from tqdm.auto import tqdm
import random


label2color = [[59, 238, 119], [222, 21, 229], [94, 49, 164], [206, 221, 133], [117, 75, 3],
                 [210, 224, 119], [211, 176, 166], [63, 7, 197], [102, 65, 77], [194, 134, 175],
                 [209, 219, 50], [255, 44, 47], [89, 125, 149], [110, 27, 100]]

labels =  [
            "__ignore__",
            "Aortic_enlargement",
            "Atelectasis",
            "Calcification",
            "Cardiomegaly",
            "Consolidation",
            "ILD",
            "Infiltration",
            "Lung_Opacity",
            "Nodule/Mass",
            "Other_lesion",
            "Pleural_effusion",
            "Pleural_thickening",
            "Pneumothorax",
            "Pulmonary_fibrosis"
            ]

viz_labels = labels[1:]

def plot_img(img, figsize=(18, 18), is_rgb=True, title="", cmap='gray'):
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
    plt.show()


def draw_bbox(image, box, label, color, thickness=3):
    alpha = 0.1
    alpha_box = 0.4
    overlay_bbox = image.copy()
    overlay_text = image.copy()
    output = image.copy()

    text_width, text_height = cv2.getTextSize(label.upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    cv2.rectangle(overlay_bbox, (box[0], box[1]), (box[2], box[3]),
                color, -1)
    cv2.addWeighted(overlay_bbox, alpha, output, 1 - alpha, 0, output)
    cv2.rectangle(overlay_text, (box[0], box[1]-7-text_height), (box[0]+text_width+2, box[1]),
                (0, 0, 0), -1)
    cv2.addWeighted(overlay_text, alpha_box, output, 1 - alpha_box, 0, output)
    cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]),
                    color, thickness)
    cv2.putText(output, label.upper(), (box[0], box[1]-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def plot_imgs(data, cols=3, size=10, is_rgb=True, title="", cmap='gray', img_size=None, thickness=3):
    if isinstance(data, pd.core.frame.DataFrame):
        df = data
        imgs = []
        for img_id, path in zip(df['image_id'], df['image_path']):

            boxes = df.loc[df['image_id'] == img_id, ['x_min', 'y_min', 'x_max', 'y_max']].values
            img_labels = df.loc[df['image_id'] == img_id, ['class_id']].values
            if len(img_labels) == 1:
                img_labels = img_labels.squeeze(axis=0)
            else:
                img_labels = img_labels.squeeze()

            img = cv2.imread(path)
            
            for label_id, box in zip(img_labels, boxes):
                color = label2color[label_id]
                img = draw_bbox(img, list(np.int_(box)), viz_labels[label_id], color, thickness=thickness)
            imgs.append(img)
    else:
        imgs = data

    rows = len(imgs)//cols + 1
    fig = plt.figure(figsize=(cols*size, rows*size))
    for i, img in enumerate(imgs):
        if img_size is not None:
            img = cv2.resize(img, img_size)
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
    

def main():
    # test plot_images
    df = pd.read_csv('train.csv')
    img_source = 'train'
    df['image_path'] = df.image_id.apply(lambda x: os.path.join(img_source, x + '.png'))
    plot_imgs(df[df.class_id != 14][:5])


if __name__ == '__main__':
    main()
