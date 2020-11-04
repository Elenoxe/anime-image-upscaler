import os
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, resize
from torchvision.utils import save_image
from tqdm import tqdm

from data import get_image_paths
from models import IMDN
from utils import ImageSplitter

checkpoints = [
    'checkpoints/IMDN-2x-level0-38.70(0.9985).pth',
    'checkpoints/IMDN-2x-level1-34.57(0.9918).pth',
    'checkpoints/IMDN-2x-level2-32.66(0.9870).pth'
]

patch_size = 128
scale_factor = 2

model = IMDN(scale_factor=scale_factor)
model.eval()
model.cuda()


def generate_test_images(src, dest, filename, size=300, noises=None):
    if noises is None:
        noises = [0, 15, 25, 35, 50]
    img: Image.Image = Image.open(src).convert('RGB')
    img = resize(img, size, Image.LANCZOS)
    for noise in noises:
        quality = 100 - noise
        img.save(os.path.join(dest, f'noise-{noise}', filename), format='JPEG', quality=quality)


images = get_image_paths('test/original')

for img_path in tqdm(images):
    relative_path = os.path.relpath(img_path, 'test/original')

    img: Image.Image = Image.open(img_path).convert('RGB')
    img_tensor = to_tensor(img).cuda()
    with torch.no_grad():
        img_tensors_up = []
        bicubic = to_tensor(resize(img, (img.height * scale_factor, img.width * scale_factor), Image.BICUBIC)).cuda()
        img_tensors_up.append(bicubic)
        for checkpoint in checkpoints:
            model.load_state_dict(torch.load(checkpoint))
            splitter = ImageSplitter()
            img_tensor_up = splitter.rescale(model, img_tensor)
            img_tensors_up.append(img_tensor_up)

        save_path = os.path.join('test/results', relative_path)
        dirname, _ = os.path.split(save_path)
        os.makedirs(dirname, exist_ok=True)
        save_image(img_tensors_up, save_path, nrow=2)
