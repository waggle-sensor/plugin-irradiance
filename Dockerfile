FROM waggle/plugin-opencv:4.1.1

COPY requirements.txt /app/
RUN pip3 --no-cache-dir install -r /app/requirements.txt

WORKDIR /app
COPY cloud_segmentation.py fcn_resnet101.pth.tar /app/
COPY models /app/models

ENV WAGGLE_PLUGIN_ID="101" \
    WAGGLE_PLUGIN_VERSION="0.1.0" \
    WAGGLE_PLUGIN_NAME="fcn_cloud_plugin" \
    WAGGLE_PLUGIN_REF="https://github.com/waggle-sensor/plugin-fcn-cloud"
    SAGE_HOST="https://sage-storage-api.nautilus.optiputer.net"

RUN sage-cli.py storage files download  e07607b5-cb20-492a-b377-e37cf3b79e4f  fcn_resnet101.pth.tar --target fcn_resnet101.pth.tar

ENTRYPOINT ["/usr/bin/python3", "/app/cloud_segmentation.py"]
