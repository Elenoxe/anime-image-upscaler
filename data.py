import os
import random

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def get_image_paths(root, extensions=IMG_EXTENSIONS):
    images = []
    for path, dirs, files in os.walk(root):
        for filename in files:
            _, ext = os.path.splitext(filename)
            if ext in extensions:
                fullname = os.path.join(path, filename)
                images.append(fullname)
    return images


def merge_image_folder(dest, *src, extensions=IMG_EXTENSIONS):
    for root in src:
        images = get_image_paths(root, extensions)
        for img in tqdm(images):
            relative_path = os.path.relpath(img, root)
            os.renames(img, os.path.join(dest, relative_path))


def split_image_folder(src, train_folder, test_folder, extensions=IMG_EXTENSIONS, train_size=None, test_size=None):
    images = get_image_paths(src, extensions)
    images_train, images_test = train_test_split(images, train_size=train_size, test_size=test_size)
    for img in tqdm(images_train):
        relative_path = os.path.relpath(img, src)
        os.renames(img, os.path.join(train_folder, relative_path))
    for img in tqdm(images_test):
        relative_path = os.path.relpath(img, src)
        os.renames(img, os.path.join(test_folder, relative_path))


class ImageData(Dataset):
    def __init__(self, images, patch_size, scale_factor,
                 preprocess=None, transform=None, target_transform=None,
                 random_crop=True, interpolation=None):
        self.images = images
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.preprocess = preprocess
        self.transform = transform
        self.target_transform = target_transform
        self.random_crop = random_crop
        self.interpolation = interpolation

    def crop(self, img):
        crop = (transforms.RandomCrop if self.random_crop else transforms.CenterCrop)(
            self.patch_size * self.scale_factor)
        return crop(img)

    def downsample(self, img):
        interpolation = self.interpolation
        if interpolation is None:
            interpolation = random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS])
        return transforms.Resize(self.patch_size, interpolation)(img)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')

        if self.preprocess:
            self.preprocess(img)

        target = self.crop(img)
        img = self.downsample(target)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)


class ImageFolder(ImageData):
    def __init__(self, root, patch_size, scale_factor,
                 preprocess=None, transform=None, target_transform=None,
                 random_crop=True, interpolation=None,
                 extensions=IMG_EXTENSIONS):
        self.root = root
        self.images = get_image_paths(root, extensions)
        super(ImageFolder, self).__init__(self.images, patch_size, scale_factor,
                                          preprocess, transform, target_transform,
                                          random_crop, interpolation)
