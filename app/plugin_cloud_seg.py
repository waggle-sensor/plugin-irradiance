import sys
import os
import argparse

import torch
from torchvision import transforms
from PIL import Image

import cv2
import numpy as np
import time
import datetime

### start plugin-hello
#!/usr/bin/env python3
# ANL:waggle-license
#  This file is part of the Waggle Platform.  Please see the file
#  LICENSE.waggle.txt for the legal details of the copyright and software
#  license.  For more details on the Waggle project, visit:
#           http://www.wa8.gl
# ANL:waggle-license
import waggle.plugin
import time
import logging

plugin = waggle.plugin.Plugin()
#### end plugin-hello

def live_feed():
    #cap = cv2.VideoCapture('http://${name}:8090/live')
    cap = cv2.VideoCapture('./image/0021.png')
    _, image = cap.read()

    return image

def run(model, args):
    if args.input == 'live':
        input_image = live_feed()
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        if args.save == True:
            filename = args.str_timestamp + '.jpg'
            cv2.imwrite(os.path.join(args.output, filename), input_image)

            outputname = args.str_timestamp + '_cloud.jpg'
            output_path = os.path.join(args.output, outputname)

        input_image = cv2.resize(input_image, (300, 300))
    else:
        #image = cv2.imread(args.input)
        input_image = Image.open(args.input)
        input_image = input_image.resize((300, 300))

        output_name = os.path.basename(args.input) + "_out.jpg"
        output_path = os.path.join(args.output, output_name)


    print(output_path)

    if type(input_image.size) != tuple:   ## --input live
        size = (300, 300)
        print(size)
    else:
        size = input_image.size
        print(size)

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


    if args.save == True:
        # create a color pallette, selecting a color for each class
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        number_of_classes = int(args.n_classes)
        colors = torch.as_tensor([i for i in range(number_of_classes)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")
        # plot the semantic segmentation predictions of 21 classes in each color
        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(size)
        #r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
        r.putpalette(colors)
        r.convert('RGB').save(output_path)


    ## calculate ratio
    np_output = np.asarray(r)
    cloud = 0
    for i in range(len(output_predictions)):
        for j in range(len(output_predictions[0])):
            if np_output[i][j] == 1:  # cloud
                cloud += 1

    total = output_predictions.shape[0] * output_predictions.shape[1]
    ratio = round((cloud / total), 5)

    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--fcn', type=str, default='101', help='FCN layer: choose fcn101 or fcn50')
    parser.add_argument('--model', type=str, default='model_best.pth.tar', help='Model path')
    parser.add_argument('--n_classes', type=int, default=2, help='Total number of classes')
    parser.add_argument('--mode', type=str, default='test', help='Purpose of running this script: [test, profile, deploy]')
    parser.add_argument('--input', type=str, help='Input image path')

    parser.add_argument('--save', type=bool, default=False, help='Save the output images? Any parameter is noticed as True')   ### --save false --> True because there is a argument
    parser.add_argument('--output', type=str, default='/output', help='Ouput folder path')

    parser.add_argument('--interval', type=int, default=15, help='Time interval of each processing')     ### With regard to the requirement of Science problem

    args = parser.parse_args()

    args.backbone = 'resnet'


    if args.save == True:
        timestamp = datetime.datetime.now()
        timestamp = timestamp.astimezone(datetime.timezone(datetime.timedelta(0)))
        args.str_timestamp = timestamp.strftime('%Y-%m-%dT%H:%M:%S%z')  ## UTC offset in the form +-HHMM (%z option in datetime)

    if args.input == None:
        parser.print_help()
        exit(0)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    from importlib import import_module
    model_module = import_module('models.{}.fcn{}'.format(args.backbone, args.fcn))
    model = model_module.FCN(n_class=int(args.n_classes))

    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['model_state_dict'])


    if args.mode == 'profile':
        try:
            while True:
                start = time.time()
                run(model, args)
                end = time.time()
                print(1 / (end - start), 'fps')
        except KeyboardInterrupt:
            print('End run')
            exit(0)
    elif args.mode == 'test':
        value = run(model, args)
        print('result:', value)
    elif args.mode == 'deploy':
        while True:
            value = run(model, args)

            plugin.add_measurement({
                'sensor_id': 0x3002,
                'parameter_id': 1,
                'value': value,
            })
            print('publish', flush=True)
            plugin.publish_measurements()

            time.sleep(args.interval)
