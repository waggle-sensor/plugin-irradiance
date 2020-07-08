FROM waggle/plugin-opencv:4.1.1-cuda

COPY plugin /plugin

RUN pip3 --no-cache-dir install -r /plugin/requirements.txt

WORKDIR /plugin

ENV WAGGLE_PLUGIN_ID="101" \
    WAGGLE_PLUGIN_VERSION="0.1.0" \
    WAGGLE_PLUGIN_NAME="fcn_cloud_plugin" \
    WAGGLE_PLUGIN_REF="https://github.com/waggle-sensor/plugin-fcn-cloud"

CMD ["/usr/bin/python3", "/plugin/plugin_bin/plugin_node"]
