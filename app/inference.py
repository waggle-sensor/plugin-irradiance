import sys
import os
import os.path as osp
import argparse

import torch
from torchvision import transforms
from PIL import Image

import json
from types import SimpleNamespace


def run(file_path, model, cfg):

    print(cfg)

    output_base = '/storage/' + cfg['output_dir']
    if not osp.exists(output_base):
        os.makedirs(output_base)

    output_name = os.path.basename(file_path) + "_out.jpg"
    output_path = os.path.join(output_base, output_name)

    print(output_path)

    input_image = Image.open(file_path)
    input_image = input_image.resize((300, 300))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
       input_batch = input_batch.to('cuda')
       model.to('cuda')
    with torch.no_grad():
        output = model(input_batch)[0]

    output_predictions = output.argmax(0)


    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    number_of_classes = int(cfg['n_classes'])
    colors = torch.as_tensor([i for i in range(number_of_classes)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)
    r.convert('RGB').save(output_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='config.list', help='path to inference configuration list')
    parser.add_argument('--image', type=str, help='path to image for inference')
    args = parser.parse_args()

    if args.image == None:
        parser.print_help()
        exit(0)

    config_path = '/storage/' + args.config
    with open(config_path, 'r') as f:
        configurations = json.load(f)

    opts = SimpleNamespace()
    opts.cfg = configurations


    from importlib import import_module
    model_module = import_module('models.{}.fcn{}'.format(opts.cfg['backbone'], opts.cfg['fcn']))
    model = model_module.FCN(n_class=int(opts.cfg['n_classes']))

    model_path = '/storage/' + opts.cfg['model']
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])


    image_path = '/storage/data/' + args.image
    run(image_path, model, opts.cfg)

