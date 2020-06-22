import sys
import os
import os.path as osp
import argparse

import torch
from torchvision import transforms
from PIL import Image


def run(model, args):

    if not osp.exists(args.output):
        os.makedirs(args.output)

    output_name = os.path.basename(args.input) + "_out.jpg"
    output_path = os.path.join(args.output, output_name)

    print(output_path)

    input_image = Image.open(args.input)
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
    number_of_classes = int(args.n_classes)
    colors = torch.as_tensor([i for i in range(number_of_classes)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)
    r.convert('RGB').save(output_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--fcn', type=str, default='101', help='FCN layer: choose fcn101 or fcn50')
    parser.add_argument('--output', type=str, default='./output', help='Ouput folder path')
    parser.add_argument('--model', type=str, default='model_best.pth.tar', help='Model path')
    parser.add_argument('--n_classes', type=int, default=2, help='Total number of classes')
    parser.add_argument('--input', type=str, help='Input image path')
    parser.add_argument('--mode', type=str, help='Purpose of running this script: [test, profile, deploy]')

    args = parser.parse_args()

    args.backbone = 'resnet'

    if args.input == None:
        parser.print_help()
        exit(0)

    from importlib import import_module
    model_module = import_module('models.{}.fcn{}'.format(args.backbone, args.fcn))
    model = model_module.FCN(n_class=int(args.n_classes))

    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['model_state_dict'])

    run(model, args)

