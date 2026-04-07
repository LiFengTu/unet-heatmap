from torch.utils.data import Dataset, DataLoader
import os
import random
import shutil
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

def ResetHeatmapYOLOFolder(data_dir, train_ratio=0.8):
    for file in os.listdir(data_dir):
        class_path = data_dir+"/"+file
        
        if os.path.isdir(class_path):
            print('class name: '+ file)
            train_path = class_path+'/train'
            if not os.path.isdir(train_path):
                os.makedirs(train_path)
            
            val_path = class_path+'/val'
            if not os.path.isdir(val_path):
                os.makedirs(val_path)

            included_extensions = ['jpg','jpeg', 'png', 'tif', 'tiff']
            file_names = [fn for fn in os.listdir(class_path)
                            if any(fn.endswith(ext) for ext in included_extensions)]
            
            file_count = len(file_names)
            train_count = int(len(file_names) * train_ratio)
            print(f'train images: {train_count}')
            train_samples = []
            train_samples = random.sample(file_names, int(train_count))

            for file in train_samples:
                image_source_path = class_path+'/'+file
                image_dest_path = train_path+'/'+file

                try:
                    shutil.move(image_source_path, image_dest_path)
                    #print(f"Moved file from {image_source_path} to {image_dest_path}")
                except OSError as e:
                    print(f"Error moving file: {e}")
                    print("Consider using 'shutil.move()' if source and destination are on different disks.")

                annotation_source_path = class_path+'/'+file.split(".")[0]+'.txt'
                annotation_dest_path = train_path+'/'+file.split(".")[0]+'.txt'

                try:
                    shutil.move(annotation_source_path, annotation_dest_path)
                    #print(f"Moved file from {annotation_source_path} to {annotation_dest_path}")
                except OSError as e:
                    print(f"Error moving file: {e}")
                    print("Consider using 'shutil.move()' if source and destination are on different disks.")
            
            val_names = [fn for fn in os.listdir(class_path)
                            if any(fn.endswith(ext) for ext in included_extensions)]
            
            print(f'val images: {len(val_names)}')
            for file in val_names:
                image_source_path = class_path+'/'+file
                image_dest_path = val_path+'/'+file

                try:
                    shutil.move(image_source_path, image_dest_path)
                    #print(f"Moved file from {image_source_path} to {image_dest_path}")
                except OSError as e:
                    print(f"Error moving file: {e}")
                    print("Consider using 'shutil.move()' if source and destination are on different disks.")

                annotation_source_path = class_path+'/'+file.split(".")[0]+'.txt'
                annotation_dest_path = val_path+'/'+file.split(".")[0]+'.txt'

                try:
                    shutil.move(annotation_source_path, annotation_dest_path)
                    #print(f"Moved file from {annotation_source_path} to {annotation_dest_path}")
                except OSError as e:
                    print(f"Error moving file: {e}")
                    print("Consider using 'shutil.move()' if source and destination are on different disks.")
                
# ====================== 2. Dataset (YOLO 標註 → Heatmap) ======================
class HeatmapYOLODataset(Dataset):
    def __init__(self, data_dir, sigma=10, input_size=512, phase='train'):
        #self.img_dir = img_dir
        #self.label_dir = label_dir
        self.data_dir = data_dir
        self.sigma = sigma
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
        self.phase = phase

        train_folders = []
        val_folders = []
        
        for file in os.listdir(data_dir):
            class_path = data_dir+"/"+file
            if os.path.isdir(class_path):
                class_train = class_path+"/train"
                if not os.path.isdir(class_train):
                    print(class_train+ ' does NOT exist')
                    exit()
                print(class_train)
                train_folders.append(class_train)
                
                class_val = class_path+"/val"
                if not os.path.isdir(class_val):
                    print(class_val+ ' does NOT exist')
                    exit()
                print(class_val)
                val_folders.append(class_val)
        
        self.img_paths = []
        source_folders = []
        
        if phase == 'train':
            source_folders = train_folders
        else:
            source_folders = val_folders
         
        for folder in source_folders:
            for name in os.listdir(folder) :
                if name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.img_paths .append(folder+'/'+name)
                    
        print(str(len(self.img_paths)))
        
        '''    
        self.img_paths = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        '''
        # 資料增強
        if phase == 'train':
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5),
                A.Resize(height=self.input_size[0], width=self.input_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(height=self.input_size[0], width=self.input_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def _load_yolo_labels(self, label_path, img_w, img_h):
            bboxes = []
            classes = []

            if not os.path.exists(label_path):
                return bboxes, classes

            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                cls, xc, yc, w, h = map(float, line.strip().split())

                # YOLO → pixel
                x1 = (xc - w/2) * img_w
                y1 = (yc - h/2) * img_h
                x2 = (xc + w/2) * img_w
                y2 = (yc + h/2) * img_h

                bboxes.append([x1, y1, x2, y2])
                classes.append(int(cls))

            return bboxes, classes
    
    def _draw_gaussian(self, heatmap, cx, cy, sigma):
        h, w = heatmap.shape

        tmp = np.zeros((h, w), dtype=np.float32)

        if 0 <= cx < w and 0 <= cy < h:
            tmp[cy, cx] = 1

        tmp = cv2.GaussianBlur(tmp, (0, 0), sigma)

        if tmp.max() > 0:
            tmp /= tmp.max()

        return np.maximum(heatmap, tmp)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        #base_name = os.path.splitext(os.path.basename(img_path))[0]
        base_name = img_path.split(".")[0]
        label_path = base_name + '.txt'
        print(label_path)
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img.shape[:2]
        
        bboxes, classes = self._load_yolo_labels(label_path, orig_w, orig_h)
        
        # Heatmap
        heatmap = np.zeros((2, orig_h, orig_w), dtype=np.float32)  # 2 classes

        for box, cls in zip(bboxes, classes):
            x1, y1, x2, y2 = box
            
            cx = int((x1 + x2) / 2 )
            cy = int((y1 + y2) / 2 )

            box_w = x2 - x1
            sigma = box_w / 6
            
            #self._draw_gaussian(heatmap[cls], (center_x, center_y), radius)
            heatmap[cls] = self._draw_gaussian(heatmap[0], (center_x, center_y), sigma)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        img = torch.tensor(img, dtype=torch.float32)
        heatmap = torch.tensor(heatmap, dtype=torch.float32)

        return img, heatmap
