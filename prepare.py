import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import glob
import yaml
import shutil
from sklearn.model_selection import StratifiedKFold
from ensemble_boxes import *
from path import Path
from plots import plot_imgs
import cv2
from collections import Counter
import argparse
from plots import *

label2color = [[59, 238, 119], [222, 21, 229], [94, 49, 164], [206, 221, 133], [117, 75, 3],
                 [210, 224, 119], [211, 176, 166], [63, 7, 197], [102, 65, 77], [194, 134, 175],
                 [209, 219, 50], [255, 44, 47], [89, 125, 149], [110, 27, 100]]

def add_features(df):
    df['x_min'] = df.x_min/df.width
    df['y_min'] = df.y_min/df.height
    df['x_max'] = df.x_max/df.width
    df['y_max'] = df.y_max/df.height

    return df


def get_cls_names(df, no_finding=False):
    class_ids, class_names = list(zip(*set(zip(df.class_id, df.class_name))))
    classes = list(np.array(class_names)[np.argsort(class_ids)])
    classes = list(map(lambda x: str(x), classes))

    return classes if no_finding else classes[:-1]


def prepare_txt(txt_path, img_list, extra_path='drive/My Drive'):
  with open(txt_path, 'w') as f:
    for f_name in tqdm(img_list):
      f.write(os.path.join('..',  extra_path, f_name) + '\n')


def prepare_yaml(save_dir, yaml_name, txt_paths, nc, cls_names, extra_path='drive/My Drive'):
    yaml_cfg = {'train': os.path.join('..', extra_path, txt_paths[0]),
              'val': os.path.join('..', extra_path, txt_paths[1]),
              'nc': nc,
              'names': cls_names}

    with open(os.path.join(save_dir, yaml_name), 'w') as f:
        yaml.dump(yaml_cfg, f, default_flow_style=False)

    with open(os.path.join(save_dir, yaml_name), 'r') as yaml_f:
        print('yaml data file: \n')
        print(yaml_f.read())
    

def stratified_kfold_split(df, k=5, heatmap=True):
  skf = StratifiedKFold(n_splits = k, shuffle = True, random_state=0)

  df_folds = df[['image_id']].copy()

  df_folds['bbox_count'] = 1
  df_folds = df_folds.groupby('image_id').count()
  #df_folds['rad_set'] = df.groupby('image_id')['rad_id'].apply(lambda x: (sorted(x.unique())))
  df_folds['class_set'] = df.groupby('image_id')['class_id'].apply(lambda x: (sorted(x.unique())))

  str_cls = df_folds['class_set'].apply(lambda x: ','.join(map(str, x)) + '-').values.astype(str)
  #str_rad = df_folds['rad_set'].apply(lambda x: ''.join(x)).values.astype(str)
  str_box = df_folds['bbox_count'].values.astype(str)

  # Preparing stratify groups
  #df_folds['stratify_group'] = np.char.add(np.char.add(str_cls, str_rad).astype(str), str_box)
  df_folds['stratify_group'] = np.char.add(str_cls, str_box)
  
  # Determining which fold the x-ray will fall in
  df_folds['fold'] = 0
  skf_split = skf.split(X = df_folds.index, y = df_folds['stratify_group'])

  for fold_number, (train_index, val_index) in enumerate(skf_split):
      df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number
      
  df_folds.reset_index(inplace = True)
  df = pd.merge(df, df_folds)

  if heatmap:
    # class_id
    temp = df.groupby(["fold", "class_id"]).agg(
        count = pd.NamedAgg("class_id", "count")).reset_index()

    temp = temp.pivot_table(index = "class_id", columns = "fold", values = "count")

    plt.figure(figsize = (20, 10))
    sns.heatmap(temp, annot = True, cmap = "YlGnBu", fmt = "g")
    plt.title("Heatmap of class distribution")

    # rad_id
    # temp = df.groupby(["fold", "rad_id"]).agg(
    #     count = pd.NamedAgg("rad_id", "count")).reset_index()

    # temp = temp.pivot_table(index = "rad_id", columns = "fold", values = "count")

    # plt.figure(figsize = (20, 10))
    # sns.heatmap(temp, annot = True, cmap = "YlGnBu", fmt = "g")
    # plt.title("Heatmap of class distribution")
  
  return df


def discard_images(df, experts=['R8', 'R9', 'R10']):
  all_rads = df[df.class_id==14].groupby('image_id').rad_id.unique()
  dis_list = all_rads[all_rads.apply(lambda x: np.all([expert not in x for expert in experts]))]
  mask = [x not in set(dis_list.index.values) for x in tqdm(df.image_id.values)]

  print('No. all class-14 images: %d' % len(all_rads))
  print('No. discarded images: %d' % len(dis_list))
  print('No. class-14 train images: %d' % (len(all_rads) - len(dis_list)))

  return df.iloc[mask], dis_list


def segregate_data(df, train_img_path, train_label_path):
  for filename in tqdm(df.image_id.unique()):
    yolo_list = []

    for _, row in df[df.image_id == filename].iterrows():
      yolo_list.append([row.class_id, row.x_cen, row.y_cen, row.w, row.h])

    yolo_list = np.array(yolo_list)
    txt_filename = os.path.join(train_label_path, filename + '.txt')
    # Save the .img & .txt files to the corresponding train and validation folders
    np.savetxt(txt_filename, yolo_list, fmt=["%d", "%f", "%f", "%f", "%f"])
    shutil.copyfile(os.path.join(row.image_path), os.path.join(train_img_path, os.path.split(row.image_path)[-1]))


def dec1_prepare(source_dir, data_save_dir, yaml_name='detect1.yaml', 
        nc=14, experts=['R8', 'R9', 'R10'], k=5, heatmap=False, ext='.png', txt_prefix='dec1_', extra_path='drive/My Drive'):
    df = pd.read_csv(os.path.join(source_dir, 'train.csv'))

    print('discarding images ...')
    df, dis_list = discard_images(df, experts)
    print('done discarding images ^^')
    print()

    print('adding features ...')
    df = add_features(df)
    print('done adding features ^^')
    print()
    
    class_names = get_cls_names(df)
    df['image_path'] = df.image_id.apply(lambda x: os.path.join(source_dir, 'train',  x + ext))

    print('fusing boxes ...')
    fused_df = fusing_boxes(df)
    print('done fusing boxes ^^')
    print()

    print('adding features ...')
    fused_df = add_xywh(fused_df)
    fused_df['image_path'] = fused_df.image_id.apply(lambda x: os.path.join(source_dir, 'train',  x + ext))
    print('done adding features ^^')
    print()

    print('kfold spliting ...')
    fused_df = stratified_kfold_split(fused_df, k=k, heatmap=heatmap)
    print('done kfold spliting ^^')
    print()

    fold = 0
    df_train = fused_df[fused_df.fold != fold]
    df_val = fused_df[fused_df.fold == fold]

    img_train_dir = os.path.join(data_save_dir, 'images', 'train')
    img_val_dir = os.path.join(data_save_dir, 'images', 'val')
    label_train_dir = os.path.join(data_save_dir, 'labels', 'train')
    label_val_dir = os.path.join(data_save_dir, 'labels', 'val')

    os.makedirs(img_train_dir, exist_ok=True)
    os.makedirs(img_val_dir, exist_ok=True)
    os.makedirs(label_train_dir, exist_ok=True)
    os.makedirs(label_val_dir, exist_ok=True)

    print('segregating data ...')
    segregate_data(df_train, img_train_dir, label_train_dir)
    segregate_data(df_val, img_val_dir, label_val_dir)
    print('done segregating data ^^')
    print()
    
    df_train['image_new_path'] = df_train.image_id.apply(lambda x: os.path.join(img_train_dir, x + ext))
    df_val['image_new_path'] = df_val.image_id.apply(lambda x: os.path.join(img_val_dir, x + ext))

    print('preparing .yaml ...')
    train_txt = os.path.join(data_save_dir, txt_prefix + 'train.txt')
    val_txt = os.path.join(data_save_dir, txt_prefix + 'val.txt')

    prepare_txt(train_txt, df_train.image_new_path.unique(), extra_path=extra_path)
    prepare_txt(val_txt, df_val.image_new_path.unique(), extra_path=extra_path)

    prepare_yaml(data_save_dir, yaml_name, (train_txt, val_txt), nc, class_names, extra_path=extra_path)

    return df_train, df_val, dis_list


def fusing_boxes(df, fusing_function='wbf', iou_thr=0.5, skip_box_thr=0.0001, sigma=0.1, verbose=False, plot_images=False, cols=2):
    assert fusing_function in {'wbf', 'mns', 'soft_mns', 'non_max_weighted'}, 'invalid fusing function'
    new_df = pd.DataFrame(columns=['image_id', 'class_id', 'x_min', 'y_min', 'x_max', 'y_max'])
    viz_images = []
    
    imagepaths = df.image_path.unique()
    
    for i, path in tqdm(enumerate(imagepaths)):
        image_id = os.path.split(path)[-1].split('.')[0]
        img_array  = cv2.imread(path)
        image_basename = Path(path).stem
        img_annotations = df[df.image_id==image_basename]

        if plot_images:
            boxes_viz = img_annotations[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().tolist()
            labels_viz = img_annotations['class_id'].to_numpy().tolist()

            ## Visualize Original Bboxes
            img_before = img_array.copy()
            for box, label in zip(boxes_viz, labels_viz):
                x_min, y_min, x_max, y_max = (box[0], box[1], box[2], box[3])
                color = label2color[int(label)]
                img_before = draw_bbox(img_before, list(np.int_(box)), viz_labels[label], color)
            viz_images.append(img_before)

        if verbose:
            print(f"(\'{image_basename}\', \'{path}\')")
            print("Bboxes before nms:\n", boxes_viz)
            print("Labels before nms:\n", labels_viz)

        boxes_list = []
        scores_list = []
        labels_list = []
        weights = []

        boxes_single = []
        labels_single = []

        cls_ids = img_annotations['class_id'].unique().tolist()
        count_dict = Counter(img_annotations['class_id'].tolist())

        for cid in cls_ids:
            ## Performing Fusing operation only for multiple bboxes with the same label
            if count_dict[cid]==1:
                labels_single.append(cid)
                boxes_single.append(img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().squeeze().tolist())

            else:
                cls_list =img_annotations[img_annotations.class_id==cid]['class_id'].tolist()
                labels_list.append(cls_list)
                bbox = img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()
                ## Normalizing Bbox by Image Width and Height
                bbox = bbox/(img_array.shape[1], img_array.shape[0], img_array.shape[1], img_array.shape[0])
                bbox = np.clip(bbox, 0, 1)
                boxes_list.append(bbox.tolist())
                scores_list.append(np.ones(len(cls_list)).tolist())

                weights.append(1)

        # Perform WBF
        if fusing_function == 'wbf':
            boxes, scores, box_labels= weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, \
                    iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        # Perform NMS
        elif fusing_function == 'nms':
            boxes, scores, box_labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
        # Perform Soft-NMS
        elif fusing_function == 'soft_nms':
            boxes, scores, box_labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, \
                    iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
        # Perform Non-maximum Weighted
        else:
            boxes, scores, box_labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, \
                    iou_thr=iou_thr,skip_box_thr=skip_box_thr)

        boxes = boxes*(img_array.shape[1], img_array.shape[0], img_array.shape[1], img_array.shape[0])
        boxes = boxes.round(1).tolist()
        box_labels = box_labels.astype(int).tolist()

        boxes.extend(boxes_single)
        boxes = np.array(boxes)
        box_labels.extend(labels_single)

        new_df = new_df.append(pd.DataFrame({'image_id': [image_id]*len(boxes), 'class_id': box_labels, \
                'x_min': boxes[:, 0], 'y_min': boxes[:, 1], 'x_max': boxes[:, 2], 'y_max': boxes[:, 3]}))

        if verbose:
            print("Bboxes after nms:\n", boxes)
            print("Labels after nms:\n", box_labels)

        ## Visualize Bboxes after operation
        if plot_images:
            img_after = img_array.copy()
            for box, label in zip(boxes, box_labels):
                color = label2color[int(label)]
                img_after = draw_bbox(img_after, list(np.int_(box)), viz_labels[label], color)
            viz_images.append(img_after)
            print()

    if plot_images:
        plot_imgs(viz_images, cols=cols, cmap=None)
        plt.figtext(0.3, 0.9,"Original Bboxes", va="top", ha="center", size=25)
        plt.figtext(0.73, 0.9,"Non-max Suppression", va="top", ha="center", size=25)
        plt.savefig('nms.png', bbox_inches='tight')
        plt.show()

    return new_df


def get_img_size(img_list):
    imgs, widths, heights = [], [], []

    for img in img_list:
        array = cv2.imread(img)
        widths.append(array.shape[1])
        heights.append(array.shape[0])
        imgs.append(os.path.split(img)[-1])

    return imgs, widths, heights


# def prepare_nih_full_data_csv(csv_path, image_source_dir):
#     df = pd.read_csv(csv_path)
#     for i, v in enumerate(df_one_hot['Finding Labels'].values):
#     v = v.split('|')
#     for c in v:
#         df_one_hot.loc[i , c] = 1
# 
# 
#     # df = df_one_hot[['Image Index', 'Cardiomegaly', 'Emphysema', 'Effusion', 'No Finding', 'Hernia',\
#     #    'Infiltration', 'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax',\
#     #    'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema',\
#     #    'Consolidation']].fillna(0)
# 
#     df = df_one_hot[['Image Index', 'Cardiomegaly', 'Emphysema', 'Effusion', 'No Finding', 'Hernia',\
#        'Infiltration', 'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax',\
#        'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema',\
#        'Consolidation']].fillna(0)
# 
#     return new_df


def prepare_nih_bbox_csv(csv_path, image_source_dir, class_map):
    df = pd.read_csv(csv_path)

    df['image_id'] = df['Image Index'].apply(lambda x: x.split('.')[0])
    df['class_id'] = df['Finding Label'].map(class_map)
    df = df[df.class_id.notnull()]
    df['class_id'] = df.class_id.astype('int8')

    df['image_path'] = df['Image Index'].apply(lambda x: os.path.join(image_source_dir, x))
    
    # _, widths, heights = get_img_size(df.image_path)
    # df['width'] = widths
    # df['height'] = heights
    df['x_cen'] = df['Bbox [x']/df.width
    df['y_cen'] = df['y']/df.height
    df['w'] = df['w']/df.width
    df['h'] = df['h]']/df.height

    df = df[['image_id', 'class_id', 'x_cen', 'y_cen', 'w', 'h', 'width', 'height', 'image_path']]
    df['source'] = 'nih'
    df['area'] = df['w']*df['h']
    return df


def prepare_dect2(csv_path, image_source_dir, data_save_dir, df_train_vin, df_val_vin,\
        k=5, heatmap=True, txt_prefix='dec2_', yaml_name='detect2.yaml', nc=6, ext='.png'):

    class_names = ['Atelectasis', 'Cardiomegaly', 'Infiltration', 'Nodule/Mass', 'Pleural effusion', 'Pneumothorax']
    
    class_map = dict(Atelectasis = 0, Cardiomegaly = 1, Infiltrate = 2, Mass = 3, Nodule = 3, Effusion = 4, Pneumothorax = 5)

    # adding features to the dataframe
    print('preparing csv file ...')
    df = prepare_nih_bbox_csv(csv_path, image_source_dir, class_map)
    print('done preparing ^^')
    print()
    
    # spliting train / val dataset
    print('spliting train val dataset ...')
    df = stratified_kfold_split(df, k=k, heatmap=heatmap)
    print('done spliting data ^^')
    print()
     
    fold = 0
    df_train = df[df.fold != fold]
    df_val = df[df.fold == fold]

    # preparing train / val dirs
    img_train_dir = os.path.join(data_save_dir, 'images', 'train')
    img_val_dir = os.path.join(data_save_dir, 'images', 'val')
    label_train_dir = os.path.join(data_save_dir, 'labels', 'train')
    label_val_dir = os.path.join(data_save_dir, 'labels', 'val')

    os.makedirs(img_train_dir, exist_ok=True)
    os.makedirs(img_val_dir, exist_ok=True)
    os.makedirs(label_train_dir, exist_ok=True)
    os.makedirs(label_val_dir, exist_ok=True)

    # copying images to the appropriate dirs
    # creating .txt labels files
    print('segregating data ...')
    segregate_data(df_train, img_train_dir, label_train_dir)
    segregate_data(df_val, img_val_dir, label_val_dir)
    print('done segregating data ^^')
    print()

    df_train['image_new_path'] = df.image_id.apply(lambda x: os.path.join(img_train_dir, x + ext))
    df_val['image_new_path'] = df.image_id.apply(lambda x: os.path.join(img_val_dir, x + ext))

    print('filtering vin data ...')
    df_train_vin = filter_vin_to_nih(df_train_vin)
    df_val_vin = filter_vin_to_nih_df(df_val_vin)
    print('done filtering vin data ^^')
    print()

    # concatenate each pair of dataframes
    print('concatenating dataframes ...')
    df_train = pd.concatenate([df_train, df_train_vin], axis=0, ignore_index=True) 
    df_val = pd.concatenate([df_val, df_val_vin], axis=0, ignore_index=True)
    print('done concatenating ^^')
    print()

    # prepare .txt files
    train_txt = os.path.join(data_save_dir, txt_prefix + 'train.txt')
    val_txt = os.path.join(data_save_dir, txt_prefix + 'val.txt')

    print('preparing .txt files ...')
    prepare_txt(train_txt, df_train.image_new_path.unique())
    prepare_txt(val_txt, df_val.image_new_path.unique())
    print('done preparing .txt files ^^')
    print()
    
    # prepare .yaml file
    print('preparing .yaml files ...')
    prepare_yaml(data_save_dir, yaml_name, (train_txt, val_txt), nc, class_names)

    return df_train, df_val


def filter_vin_to_nih(df):
    change_id_map = {1:0, 3:1, 6:2, 8:3, 10:4, 12:5, 14:6}
    df['class_id'] = df.class_id.map(change_id_map)
    df = df[df.class_id.notnull()]
    df.class_id = df.class_id.astype('int8')
    df['source'] = 'vin'

    return df

# def prepare_nih(bbox_csv_path, ):
#     df_vin
#     df_nih
#     30% of size(df_nih) from nofinding of df_nih_full
#     30% of size(df_vin) from nofinding of df_nih_full
#     filter out the valid class for df_vin
#     split train val df_vin
#     split train val df_nih

def add_xxyy(df):
    df['x_min'] = df.apply(lambda row: row.x_cen - row.w/2, axis=1)
    df['y_min'] = df.apply(lambda row: row.y_cen - row.h/2, axis=1)
    df['x_max'] = df.apply(lambda row: row.x_cen + row.w/2, axis=1)
    df['y_max'] = df.apply(lambda row: row.y_cen + row.h/2, axis=1)

    return df

def add_xywh(df):
    df['x_cen'] = (df.x_min + df.x_max)/2
    df['y_cen'] = (df.y_min + df.y_max)/2
    df['w'] = df.x_max - df.x_min 
    df['h'] = df.y_max - df.y_min 

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls-names', action='store_true')
    parser.add_argument('--txt', action='store_true')
    parser.add_argument('--yaml', action='store_true')
    parser.add_argument('--kfold', action='store_true')
    parser.add_argument('--discard', action='store_true')
    parser.add_argument('--seg', action='store_true')
    parser.add_argument('--fuse', action='store_true')
    parser.add_argument('--img-size', action='store_true')
    parser.add_argument('--nih', action='store_true')
    parser.add_argument('--filter', action='store_true')
    parser.add_argument('--dec1', action='store_true')
    parser.add_argument('--dec2', action='store_true')
    parser.add_argument('--add-wh-nih', action='store_true')
    parser.add_argument('--deploy', action='store_true')
    parser.add_argument('--source', type=str)
    parser.add_argument('--save', type=str)
    parser.add_argument('--extra', type=str)

    opt = parser.parse_args()
    
    # if not opt.deploy and not opt.dec1:
    #     df_vin = pd.read_csv('train.csv')
    #     df_nih = pd.read_csv('BBox_List_2017.csv')
    #     class_names = get_cls_names(df_vin)
    #     img_list = glob.glob('../*.jpg')
    
    if opt.add_wh_nih:
        full = pd.read_csv('Data_Entry_2017.csv')
        df_nih['width'] = [full[full['Image Index']==img]['OriginalImage[Width'].values[0] for img in df_nih['Image Index'].values]
        df_nih['height'] = [full[full['Image Index']==img]['Height]'].values[0] for img in df_nih['Image Index'].values]

        print('No. left out imgs: %d' % len(df_nih[df_nih.width==0]))
        df_nih.to_csv('BBox_List_2017_.csv', header=True, index=False)

    if opt.cls_names:
        print(class_names)
        print()
        print(type(class_names))

    if opt.txt:
        prepare_txt('test_txt.txt', img_list)
        with open('test_txt.txt', 'r') as f:
            print(f.read())

    if opt.yaml:
        prepare_yaml('.', 'test_yaml.yaml', ('train.txt', 'val.txt'), 14, class_names)

    if opt.kfold:
        df_new = stratified_kfold_split(df_vin, k=5, heatmap=True)

    if opt.discard:
        df_new, dis_list = discard_images(df_vin)
        
    if opt.seg:
        segregate_data(df, 'image_test', 'label_test')

    if opt.fuse:
        img_source = 'train'
        df_vin['image_path'] = df_vin.image_id.apply(lambda x: os.path.join(img_source, x + '.png'))
        new_df_vin = fusing_boxes(df_vin[:20])
        print(new_df_vin)
        print()
        print('test class 0')
        new_df = fusing_boxes(df_vin[df_vin.class_id==0])
        print('org len class 0 = %d' % len(df_vin[df_vin.class_id==0]))
        print('new len class 0 = %d' % len(new_df[new_df.class_id==0]))

    if opt.img_size:
        imgs, widths, heights = get_img_size(img_list)
        print(imgs) 
        print(widths)
        print(heights)

    if opt.nih:
        class_map = dict(Atelectasis = 0, Cardiomegaly = 1, Infiltrate = 2, Mass = 3, Nodule = 3, Effusion = 4, Pneumothorax = 5)

        df_new = prepare_nih_bbox_csv('BBox_List_2017.csv', 'hahaha', class_map)
        print(df_new[:10])

    if opt.filter:
        new_df = filter_vin_to_nih(df_vin[:20])
        print(new_df)
        
    if opt.dec1:
        VIN_SOURCE = opt.source
        DATA_SAVE_DIR = opt.save

        os.makedirs(DATA_SAVE_DIR, exist_ok=True)
        
        print('PREPARING DETECT 1 ...') 
        print()
        df_train1, df_val1, dis_list = dec1_prepare(VIN_SOURCE, DATA_SAVE_DIR,\
                yaml_name='detect1.yaml', ext='.png', txt_prefix='dec1_', extra_path=opt.extra) 

        print('SAVING .csv FILES ...')
        CSV_DIR = os.path.join(DATA_SAVE_DIR, 'records')
        os.makedirs(CSV_DIR, exist_ok=True)

        df_train1.to_csv(os.path.join(CSV_DIR, 'train1.csv'), index=False, header=True)
        df_val1.to_csv(os.path.join(CSV_DIR, 'val1.csv'), index=False, header=True)
        dis_list.to_csv(os.path.join(CSV_DIR, 'dis_list.csv'), index=False, header=False)

    if opt.dec2:
        a = 1

    if opt.deploy:
        VIN_SOURCE = f'vin1024'
        HIU_SOURCE = f'nih1024'
        DATA_SAVE_DIR = f'vinbigdata/data'
        HIU_CSV = f'nih/BBox_List_2017.csv'

        os.makedirs(DATA_SAVE_DIR, exist_ok=True)
        
        print('PREPARING DETECT 1 ...') 
        print()
        df_train_1, df_val_1, dis_list = dec1_prepare(VIN_SOURCE, DATA_SAVE_DIR,\
                yaml_name='detect1.yaml', ext='.png', txt_prefix='dec1_') 

        print('PREPARING DETECT 2 ...')
        print()
        df_train_2, df_val_2 = prepare_dect2(HIU_CSV, HIU_SOURCE, DATA_SAVE_DIR,\
                df_train_1, df_val_1, txt_prefix='dec2_', yaml_name='detect2.yaml', ext='.png') 

        print('SAVING .csv FILES ...')
        CSV_DIR = os.path.join(DATA_SAVE_DIR, 'records')
        os.makedirs(CSV_DIR, exist_ok=True)

        df_train_1.to_csv(os.path.join(CSV_DIR, 'train1.csv'), index=False, header=True)
        df_val_1.to_csv(os.path.join(CSV_DIR, 'val1.csv'), index=False, header=True)
        df_train_2.to_csv(os.path.join(CSV_DIR, 'train2.csv'), index=False, header=True)
        df_val_2.to_csv(os.path.join(CSV_DIR, 'val2.csv'), index=False, header=True)

if __name__ == '__main__':
    main()
