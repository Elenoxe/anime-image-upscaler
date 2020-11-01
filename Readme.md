# Anime Image Upscaler
This project is inspired by [waifu2x](https://github.com/nagadomi/waifu2x).
It implements an anime image upscaler and denoiser using PyTorch.
It's a toy for those who are interested in image super resolution with deep learning,
not a tool for general users.
## Datasets
I use 6000+ random images from [Safebooru](https://safebooru.org/),
which is a safe-for-work version of [Danbooru](https://danbooru.donmai.us/).
Images are split with a ratio of 3:1 for training and validation.
Each image is at least 2k x 2k. Images are downsampled by Lanczos,
each side then have at most 2k pixels.
## Models
I trained two models on my dataset. Codes are self-written.

+ [Cascading Residual Network (CARN)](https://arxiv.org/abs/1803.08664)
+ [Information Multi-distillation Network (IMDN)](https://arxiv.org/abs/1909.11856)

For CARN I added a big residual shortcut connecting the first conv layer and the upsample layers.
This seems to improve the performance with very low cost.  
I also implemented [Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR)](https://arxiv.org/abs/1707.02921),
but it is not trained and tested due to limitation of my computing power and memory.
## Training
Original images are too large for training. They are cropped to small patches. Then downsampled to
smaller patches as input of the model. The original patches are used as training targets of the model. 
All input patches have 128 x 128 size. According to [yu45020's Waifu2x re-implementation](https://github.com/yu45020/Waifu2x),
bigger patches seem to perform better compared to 64 x 64 patches. I don't have time to test on my
model and dataset though.  
Random JPEG noises are added to input when training models with denoiser. Original inputs are encoded
to JPEG formats with quality [75, 95] for noise level 1, and [50, 75] for noise level 2. Qualities are
uniformly sampled within the ranges. Models are trained without noised input at first, then fine-tuned
for denoise ability. For now I only trained models with denoiser for IMDN, I may train a CARN version for camparision if I
have time in the feature.  
## Upscaling
Images to be upscaled should be split into patches as well. Usually same patch size as that in training is recommended.
Other sizes should be working too. The output patches are then merged into a complete image as result.