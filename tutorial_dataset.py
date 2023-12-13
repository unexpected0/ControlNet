import json
import cv2
import numpy as np
import os

from torch.utils.data import Dataset

class Movies_Dataset(Dataset):
    def __init__(self, path, file_list, standard, frame):
        self.path = path
        self.data = []
        with open (file_list,'r',encoding='utf-8') as file:
            for num, line in enumerate(file):
                line = line.strip('\n').split()
                pic_list = []
                for j in range(len(line)):
                    if float(line[j]) > standard:
                        index = num - 19 + j + int(j>=19) 
                        if -frame > num + 1 - index or num + 1 - index > frame:
                            pic_list.append(index)
                if pic_list != []:
                    file_name = str(num + 1).zfill(6) + '.png'
                    item = {'file':file_name,'list':pic_list}
                    self.data.append(item)
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        reference = cv2.imread(self.path + '/' + item['file'])
        seed = np.random.randint(0, len(item['list']))
        target = cv2.imread(self.path + '/' + str(item['list'][seed]).zfill(6) + '.png')
        
        # Do not forget that OpenCV read images in BGR order.
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

        # Normalize source images to [0, 1].
        reference = reference.astype(np.float32) / 255.0
        gray = gray.astype(np.float32) / 255.0
        gray = np.expand_dims(gray,axis=2)
        hint = np.concatenate((reference,gray), axis=2)
        
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=' ', hint=hint)
        

class Movies_Dataset_v2(Dataset):
    def __init__(self, movies_path, txt_list, low_bound, frame, upperlimit = 1):
        self.movies_path = movies_path
        self.data = []
        for txt_file in sorted(os.listdir(txt_list)):
            movie_name = txt_file.split('.')[0]
            txt_path = txt_list + '/' + txt_file
            with open (txt_path,'r',encoding='utf-8') as file:
                for num, line in enumerate(file):
                    line = line.strip('\n').split()
                    pic_list = []
                    for j in range(len(line)):
                        if float(line[j]) > low_bound and float(line[j]) < upperlimit:
                            index = num - 19 + j + int(j>=19) 
                            if -frame > num + 1 - index or num + 1 - index > frame:
                                pic_list.append(index)
                    if pic_list != []:
                        file_name = str(num + 1).zfill(6) + '.png'
                        item = {'file':file_name,'list':pic_list,'movie':movie_name}
                        self.data.append(item)
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        reference = cv2.imread(self.movies_path + '/' + item['movie'] +'/' + item['file'])
        seed = np.random.randint(0, len(item['list']))
        target = cv2.imread(self.movies_path + '/' + item['movie'] + '/' + str(item['list'][seed]).zfill(6) + '.png')
        
        # Do not forget that OpenCV read images in BGR order.
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

        # Normalize source images to [0, 1].
        reference = reference.astype(np.float32) / 255.0
        gray = gray.astype(np.float32) / 255.0
        gray = np.expand_dims(gray,axis=2)
        hint = np.concatenate((reference,gray), axis=2)
        
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=' ', hint=hint)


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        path = '/home/dailongquan/110.015/datasets/refer-color/ref'
        for file_name in os.listdir(path):
            # print(file_name)
            # exit(0)
            self.data.append(file_name)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        prompt = ' '

        reference = cv2.imread('/home/dailongquan/110.015/datasets/refer-color/ref/' + item)
        item = item.split('.')[0] + '_1.' + item.split('.')[1]
        # print(item)
        target = cv2.imread('/home/dailongquan/110.015/datasets/refer-color/target/' + item)

        reference = cv2.resize(reference,(512,512))
        target = cv2.resize(target,(512,512))
        
        
        # Do not forget that OpenCV read images in BGR order.
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

        # Normalize source images to [0, 1].
        reference = reference.astype(np.float32) / 255.0
        # print(reference.shape)
        gray = gray.astype(np.float32) / 255.0
        gray = np.expand_dims(gray,axis=2)
        # print(gray.shape)
        hint = np.concatenate((reference,gray), axis=2)
        # print(hint.shape)


        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=hint)


class MyDataset_Coco(Dataset):
    def __init__(self):
        self.data = []
        with open('/home/dailongquan/110.015/datasets/coco2017/annotations/captions_train2017.json', 'rt') as f:
            data = json.load(f)
            for annotation in data['annotations']:
                self.data.append(annotation)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image_id = item['image_id']
        caption = item['caption']
        filename = "{:012}".format(image_id)
        # print(filename)

        semantic = cv2.imread('/home/dailongquan/110.015/datasets/coco2017_seg/train2017/' + filename + '.png')
        target = cv2.imread('/home/dailongquan/110.015/datasets/coco2017/train2017/' + filename + '.jpg')
        
        semantic = cv2.resize(semantic,(512,512))
        target = cv2.resize(target,(512,512))

        # Do not forget that OpenCV read images in BGR order.
        semantic = cv2.cvtColor(semantic, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        semantic = semantic.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=caption, hint=semantic)
    
    


class MyDataset_Origin(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/fill50k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/fill50k/' + source_filename)
        target = cv2.imread('./training/fill50k/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

