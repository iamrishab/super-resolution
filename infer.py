import os
import argparse

from utils.prepare_images import *
from Models import *
from torchvision.utils import save_image

# Parse arguments
parser = argparse.ArgumentParser(description='Super Resolution GAN')
parser.add_argument('--ckpt', help='model checkpoint file',
                    default='weights/CRAN_V2/CARN_model_checkpoint.pt', type=str)
parser.add_argument('--src', help='source image path',
                    required=True, type=str)
parser.add_argument('--res', help='output image save path',
                    default='output.png', type=str)
args = parser.parse_args()


img = Image.open(args.src).convert("RGB")

model = CARN_V2(color_channels=3, mid_channels=64, conv=nn.Conv2d,
                        single_conv_size=3, single_conv_group=1,
                        scale=2, activation=nn.LeakyReLU(0.1),
                        SEBlock=True, repeat_blocks=3, atrous=(1, 1, 1))
                        
model = network_to_half(model)
if os.path.exists(args.ckpt):
    model.load_state_dict(torch.load(args.ckpt, 'cpu'))

img_t = to_tensor(img).unsqueeze(0) 

if torch.cuda.is_available():
    model = model.cuda()
    img_t = img_t.cuda()

with torch.no_grad():
    img_upscale = model(img_t)

save_image(img_upscale, args.res)
