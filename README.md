### plugin-FCN-cloud

This repository is an example to Waggle users to build their own repository containing their application and a manifest file to be running on Waggle nodes within science example of cloud coverage estimation. This repository refers [plugin-helloworld](https://github.com/waggle-sensor/plugin-helloworld). Please refer to the [README](https://github.com/waggle-sensor/plugin-helloworld/blob/master/README.md) to learn more.

### Inference Resnet-FCN network on PyTorch

The plugin inference images using fcn models: resnet101 based fcn101 or fcn50. To run the plugin, the user must have Docker engine (greater than 18.X.X) installed on the host. Nvidia CUDA driver (>= 10.1) on the host is preferrable for GPU acceleration. Additionally, the user must have [virtual waggle](https://github.com/waggle-sensor/waggle-node) to provide the script waggle environment. Please refer to the [README](https://github.com/waggle-sensor/waggle-node/blob/master/README.md) to learn more.

### Mode for the plugin

Plugins suppose to provide three modes; `test`, `profile`, and `deploy`.

`test` mode provides single run of the plugin to test the script if it works fine or has errors. <br />
`profile` mode runs the script as fast as possible infinitely to profile how much resource does the script requires. The script will stop with keyboard interruption.<br />
`deploy` mode is for deploying the script in nodes. The script will run with time interval (default is 15 seconds).

### Guidance on Testing this script

It is beneficial to use command line arguments to feed input(s) and parameters to the main code of your application. This lets the application switch between "dev" and "production" mode easily. This approach makes it very easy to register the applicaiton to our Docker registry through the Edge code repository.

To test the script with:
- the resnet101-fcn50 model
- a weight named as model_best.pth.tar
- 2 classes
- save result images in ./output
- as a test

#### on development get an image from local filesystem
```
$ cloud_segmentation.py --fcn 50 \
  --model ./model_best.pth.tar \
  --n_classes 2 \
  --input image.jpg \
  --output ./output \
  --save true --mode test 
```
#### on testing get an image from a stream of a camera
```
$ cloud_segmentation --fcn 50 \
  --model ./model_best.pth.tar \
  --n_classes 2 \
  --input http://camera:8090/live \
  --output ./output \
  --save true \
  --mode test 
```

#### Registering The Application to Edge Code Repository

The application needs to be registered in the Edge code repository to be running on Waggle nodes. [App specification](sage.json) helps to define the application manifest when registering the application. The file can directly be fed into the Edge code repository to register. However, the registration process can also be done via Waggle/SAGE UI and the app_specification.json is not required in this way.
