# Overview

 Re-implementation on the original [waifu2x](https://github.com/nagadomi/waifu2x) in PyTorch with additional super resolution models.

## Dependencies 
* Python 3x
* [PyTorch](https://pytorch.org/) >= 1
* [Nvidia/Apex](https://github.com/NVIDIA/apex/) (used for mixed precision training, you may use the [python codes](https://github.com/NVIDIA/apex/tree/master/apex/fp16_utils) directly)


## What's New
* Add [CARN Model (Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network)](https://github.com/nmhkahn/CARN-pytorch). Model Codes are adapted from the authors's [github repo](https://github.com/nmhkahn/CARN-pytorch). I add [Spatial Channel Squeeze Excitation](https://arxiv.org/abs/1709.01507) and swap all 1x1 convolution with 3x3 standard convolutions. The model is trained in fp 16 with Nvidia's [apex](https://github.com/NVIDIA/apex). Details and plots on model variant can be found in [docs/CARN](./docs/CARN)

* Dilated Convolution seems less effective (if not make the model worse) in super resolution, though it brings some improvement in image segmentation, especially when dilated rate increases and then decreases. Further investigation is needed. 


 ## Training
 
 If possible, fp16 training is preferred because it is much faster with minimal quality decrease. 
 
 Sample training script is available in `train.py`, but you may need to change some liens. 
 
 ###  Image Processing
 Original images are all at least 3k x 3K. I downsample them  by LANCZOS so that  one side has at most 2048, then I randomly cut them into 256x256 patches as target  and use 128x128 with jpeg noise as input images. All input patches have at least 14 kb, and they are stored in SQLite with BLOB format. SQlite seems to have [better performance](https://www.sqlite.org/intern-v-extern-blob.html) than file system for small objects. H5 file format may not be optimal because of its larger size. 
 
 Although convolutions can take in any sizes of images, the content of image matters. For real life images, small patches may maintain color,brightness, etc variances in small regions, but for digital drawn images, colors are added in block areas. A small patch may end up showing entirely one color, and the model has little to learn. 

Downsampling methods  are uniformly chosen among ```[PIL.Image.BILINEAR, PIL.Image.BICUBIC, PIL.Image.LANCZOS]``` , so different patches in the same image might be down-scaled in different ways. 

Image noise are from JPEG format only. They are added by re-encoding PNG images into PIL's JPEG data with various quality. Noise level 1 means quality ranges uniformly from [75, 95]; level 2 means quality ranges uniformly from [50, 75]. 
 

 ## Models
 Models are tuned and modified with extra features. 
 
 
* [DCSCN 12](https://github.com/jiny2001/dcscn-super-resolution) 

* [CRAN](https://github.com/nmhkahn/CARN-pytorch)
 
 #### From [Waifu2x](https://github.com/nagadomi/waifu2x)
 * [Upconv7](https://github.com/nagadomi/waifu2x/blob/7d156917ae1113ab847dab15c75db7642231e7fa/lib/srcnn.lua#L360)
 
 * [Vgg_7](https://github.com/nagadomi/waifu2x/blob/7d156917ae1113ab847dab15c75db7642231e7fa/lib/srcnn.lua#L334)
 
 * [Cascaded Residual U-Net with SEBlock](https://github.com/nagadomi/waifu2x/blob/7d156917ae1113ab847dab15c75db7642231e7fa/lib/srcnn.lua#L514) (PyTorch codes are not available and under testing)
 
