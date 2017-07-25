import io
from PIL import Image
import torch
from torchvision import models, transforms, datasets
from torch.autograd import Variable
import torch.utils.data as data
import numpy as np

from PIL import Image
import os
import os.path


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def load_labels_file(path):
    print(path)
    labels = {}
    tags = {'Sad':0, 'Fear':1, 'Angry':2, 'Disgust':3, 'Neutral':4, 'Happy':5, 'Surprise':6}
    with open(path,'r') as stream:
        for line in stream:
            [file_id, tag] = line.strip().split()
            labels[file_id] = tags[tag]
    return labels, tags

def make_dataset_seq(dir, class_to_idx):
    sequences = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        images = []
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
        sequences.append(images)
    return sequences


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader2(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

    
def default_loader(path):
    return Image.open(path).convert('RGB')

def imagepath_to_frame_index(path):
    filename = path.strip().split('/')[-1].split('.')[0]
    return int(filename.replace('I_1',''))-1
    

class ImageFolderSequences(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset_seq(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.pad_image = np.uint8( np.zeros([3,128,128]) )
        self.pad_image = Image.fromarray(np.rollaxis(self.pad_image, 0,3))
        if self.transform is not None:
            self.pad_image = self.transform(self.pad_image)
        self.labels, self.tags = load_labels_file(root + '/../../labels.txt')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: ([images], target) where target is class_index of the target class.
        """
        images = []
        sequence_index = 0
        for item in self.imgs[index]:
            path, target = item
            dirname = self.idx_to_class[target]
            target_class = self.labels[dirname]
#             print(dirname, target_class, target)
            target = target_class
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            frame_index = imagepath_to_frame_index(path)
            if self.target_transform is not None:
                frame_index = self.target_transform( frame_index )
            if sequence_index > 0:
                count_pad=0
                while sequence_index < frame_index: 
                    sequence_index_tmp = sequence_index
                    if self.target_transform is not None:
                        sequence_index_tmp = self.target_transform( sequence_index_tmp )
                    good_image = False
                    images.append([ self.pad_image, sequence_index_tmp, good_image])
                    sequence_index = sequence_index +1
                    count_pad=count_pad+1
                    if count_pad > 1:
                        sequence_index = frame_index
                        break

            good_image = True    
            images.append([img,frame_index, good_image])
            sequence_index = sequence_index +1

        if self.target_transform is not None:
            target = self.target_transform(target)

        return images, target

    def __len__(self):
        return len(self.imgs)
      
      
def my_collate(batch):
    _prueba_batch = batch
    max_length=0
    for n in range(len(_prueba_batch)):
        nphotos = len(_prueba_batch[n][0])
        if nphotos > max_length:
            max_length = nphotos

    data_tensor = torch.FloatTensor(
        len(_prueba_batch), 
        max_length, 
        (_prueba_batch[0][0][0][0]).size()[0], 
        (_prueba_batch[0][0][0][0]).size()[1], 
        (_prueba_batch[0][0][0][0]).size()[2]
        ).zero_()
    data_tensor.size()

    for n in range(len(_prueba_batch)):
        nphotos = len(_prueba_batch[n][0])
        #photos_tensor = torch.FloatTensor(max_length,
                                        #(_prueba_batch[0][0][0][0]).size()[0],
                                        #(_prueba_batch[0][0][0][0]).size()[1], 
                                        #(_prueba_batch[0][0][0][0]).size()[2]
                                        #).zero_()
        for p in range(nphotos):
    #        photos_tensor[p]=_prueba_batch[n][0][p][0]
#             try:
            data_tensor[n][p] = _prueba_batch[n][0][p][0]
#             except:
#                 print("n ", n)
#                 print("p ", p)
#                 print("data_tensor ", data_tensor.size())
#                 print("_prueba_batch ", len(_prueba_batch[n][0]))
#                 print("_prueba_batch ", len(_prueba_batch[n][0][p]))

    target = torch.LongTensor( len(_prueba_batch), 1).zero_()
    for n in range(len(_prueba_batch)):
        target[n] = _prueba_batch[n][1]


    return((data_tensor, target))
  
def my_collate_percentile(batch):
    _prueba_batch = batch
    max_length=0
    lengths=[]
    for n in range(len(_prueba_batch)):
        nphotos = len(_prueba_batch[n][0])
        if nphotos > max_length:
            max_length = nphotos
        lengths.append(nphotos)
#         print(nphotos)
#     print(max_length)
    median_length = np.ceil(np.median(np.asarray(lengths)))
    percentile_length = int(np.ceil(np.percentile(np.asarray(lengths), 75)))


    data_tensor = torch.FloatTensor(
        len(_prueba_batch), 
        percentile_length, 
        (_prueba_batch[0][0][0][0]).size()[0], 
        (_prueba_batch[0][0][0][0]).size()[1], 
        (_prueba_batch[0][0][0][0]).size()[2]
        ).zero_()
    data_tensor.size()

    for n in range(len(_prueba_batch)):
        nphotos = len(_prueba_batch[n][0])
        if nphotos > percentile_length:
            rest = nphotos - percentile_length
            rand_start = np.random.randint(rest)
        else:
            rand_start = np.random.randint(nphotos)
            #data_tensor[n] = _prueba_batch[n][0][rand_start:(rand_start+percentile_length)][0]
        for p in range(percentile_length):   
            data_tensor[n][p] = _prueba_batch[n][0][((p+rand_start)%nphotos + int((p+rand_start)/nphotos))%nphotos][0] #circular buffer, with one single padding image when starting over


    target = torch.LongTensor( len(_prueba_batch), 1).zero_()
    for n in range(len(_prueba_batch)):
        target[n] = _prueba_batch[n][1]


    return((data_tensor, target))
