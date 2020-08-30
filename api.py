from __future__ import print_function
import os
import json
import cv2
import uuid
import base64
import argparse
import numpy as np

from flask_cors import CORS
from flask_restplus import reqparse
from flask_restplus import Api, Resource
from werkzeug.datastructures import FileStorage
from flask import Flask, Response, jsonify, request

from utils.prepare_images import *
from Models import *
from torchvision.utils import save_image

# Parse arguments
# Parse arguments
parser = argparse.ArgumentParser(description='Super Resolution GAN')
parser.add_argument('--ckpt', help='model checkpoint file',
                    default='weights/CRAN_V2/CARN_model_checkpoint.pt', type=str)
args = parser.parse_args()


app = Flask(__name__) 
CORS(app)

api = Api(app, version='1.0', title="Lightbox APIs", validate=False, description='Image Super-Resolution')
ns = api.namespace('', description='API operations')

upload_parser = api.parser()
args_parser = reqparse.RequestParser()

upload_parser.add_argument('image', location='files', type=FileStorage, required=True)
args_parser.add_argument('filepath', required=True, help="Image File path")
args_parser.add_argument('savedir', required=True, help="Image Save path")

os.makedirs('images', exist_ok=True)

model = CARN_V2(color_channels=3, mid_channels=64, conv=nn.Conv2d,
                        single_conv_size=3, single_conv_group=1,
                        scale=2, activation=nn.LeakyReLU(0.1),
                        SEBlock=True, repeat_blocks=3, atrous=(1, 1, 1))
                        
model = network_to_half(model)
if os.path.exists(args.ckpt):
    model.load_state_dict(torch.load(args.ckpt, 'cpu'))

if torch.cuda.is_available():
    model = model.cuda()


def im2str(im_path):
    im = cv2.imread(im_path)
    _, imdata = cv2.imencode('.JPG', im)
    return base64.b64encode(imdata).decode('ascii')


@ns.route('/super')
@api.expect(args_parser)
class Segmentation(Resource):
    @ns.doc('list_todos')
    def post(self):
        args = args_parser.parse_args()
        img_path =  args["filepath"]
        save_dir = args["savedir"]
        
        os.makedirs(save_dir, exist_ok=True)
        img_prefix, ext = os.path.splitext(os.path.basename(img_path))
        
        img = Image.open(img_path).convert("RGB")
        img_t = to_tensor(img).unsqueeze(0)
        
        if torch.cuda.is_available():
            img_t = img_t.cuda()

        with torch.no_grad():
            img_upscale = model(img_t)
            
        super_res_path = os.path.abspath(os.path.join(save_dir, 'super_res.png'))
        save_image(img_upscale, super_res_path)
        response = json.dumps({"superResPath": super_res_path})
        return Response(response=response, status=200, mimetype="application/json")


@ns.route('/infer')
@api.expect(upload_parser)
class Upload(Resource):
    @ns.doc('list_todos')
    def post(self):
        args = upload_parser.parse_args()
        uploaded_file = args['image']  # This is FileStorage instance
        img_path = os.path.abspath(os.path.join('images', f'{str(uuid.uuid4())}.png'))
        hr_img_path = os.path.abspath(os.path.join('images', f'{str(uuid.uuid4())}_hr.png'))
        uploaded_file.save(img_path)

        data = os.path.abspath(os.path.dirname(img_path))
        
        img = Image.open(img_path).convert("RGB")
        img_t = to_tensor(img).unsqueeze(0)
        
        if torch.cuda.is_available():
            img_t = img_t.cuda()

        with torch.no_grad():
            img_upscale = model(img_t)
            # b, c, h, w = img_upscale.shape
            # img_upscale_np = model(img_t).squeeze(0).view(h, w, c).cpu().numpy().astype(np.uint8)
        
        save_image(img_upscale, hr_img_path)
        response = json.dumps({'result': im2str(hr_img_path)})
        os.remove(img_path)
        os.remove(hr_img_path)
        return Response(response=response, status=200, mimetype="application/json")
    

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=8003, debug=False, threaded=True)