import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from . import augmentation as Aug
from . import transform as T
from . import sketch_simplification

class CustomDataset(Dataset):
    
    def __init__(self, root_dir, transform_sketch=T.transform_sketch, transform_photo=T.transform_photo, load_photo=True, augmentation=False):
        self.load_photo = load_photo
        self.augmentation = augmentation
        
        self.transform_sketch = transform_sketch
        self.transform_photo = transform_photo
        
        self.root_dir = root_dir
        self.sketch_dir = os.path.join(root_dir, 'sketches')
        self.photo_dir = os.path.join(root_dir, 'photos')
        
        self.sketch_files = os.listdir(self.sketch_dir)
        self.photo_files = os.listdir(self.photo_dir)
    
    def __len__(self):
        return len(self.sketch_files)
            
    def __getitem__(self, idx):
        sketch = self.load_one_sketch(idx)
        
        if self.load_photo:
            photo = self.load_one_photo(idx)
            return sketch, photo
        else:
            return sketch
    
    def load_one_sketch(self, idx):
        sketch_name = self.sketch_files[idx]
        sketch_path = os.path.join(self.sketch_dir, sketch_name)
        sketch = Image.open(sketch_path)
        sketch = self.augment_sketch(sketch)
        sketch = self.transform_sketch(sketch)
        return sketch
    
    def load_one_photo(self, idx):
        photo_name = self.photo_files[idx]
        photo_path = os.path.join(self.photo_dir, photo_name)
        photo = Image.open(photo_path)
        photo = self.augment_photo(photo)
        photo = self.transform_photo(photo)
        return photo
    
    def augment_sketch(self, sketch):
        if self.augmentation:
            sketch = Aug.random_erase(sketch)
            sketch = Aug.random_affine([sketch])[0]
        return sketch
    
    def augment_photo(self, photo):
        if self.augmentation:
            photo = Aug.random_affine([photo])[0]
        return photo

def dataloader(root_dir, batch_size, load_photo=True, shuffle=True, num_workers=4, augmentation=False):
    custom_dataset = CustomDataset(root_dir, load_photo=load_photo, augmentation=augmentation)
    custom_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return custom_dataloader

simplificator = None

def load_one_sketch(path, augmentation=False, simplify=False, device='cpu'):
    sketch = Image.open(path)
    if simplify:
        global simplificator
        if not simplificator:
            simplificator = sketch_simplification.sketch_simplification(device=device)
        sketch = simplificator.simplify(sketch)
    if augmentation:
        sketch = Aug.random_erase(sketch)
        sketch = Aug.random_affine([sketch])[0]
    sketch = T.transform_sketch(sketch)
    return sketch

def load_one_photo(path, augmentation=False, simplify=False, device='cpu'):
    photo = Image.open(path)
    if simplify:
        global simplificator
        if not simplificator:
            simplificator = sketch_simplification.sketch_simplification(device=device)
        photo = simplificator.simplify(photo)
    if augmentation:
        photo = Aug.random_affine([photo])[0]
    photo = T.transform_photo(photo)
    return photo

def load_one_sketch_photo(path_sketch, path_photo, augmentation=False, simplify=False, device='cpu'):
    sketch = Image.open(path_sketch)
    photo = Image.open(path_photo)
    if simplify:
        global simplificator
        if not simplificator:
            simplificator = sketch_simplification.sketch_simplification(device=device)
        sketch = simplificator.simplify(sketch)
        photo = simplificator.simplify(photo)
    if augmentation:
        sketch = Aug.random_erase(sketch)
        sketch, photo = Aug.random_affine([sketch, photo])
    sketch = T.transform_sketch(sketch)
    photo  = T.transform_photo(photo)
    return sketch, photo
