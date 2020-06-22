FROM waggle/plugin-opencv:4.1.1-cuda

COPY /app /plugin

RUN pip3 --no-cache-dir install /plugin/requirements.txt

WORKDIR /plugin

ENV WAGGLE_PLUGIN_ID="101" \
    WAGGLE_PLUGIN_VERSION="0.1.0" \
    WAGGLE_PLUGIN_NAME="FCN Cloud Plugin" \
    WAGGLE_PLUGIN_REF="https://github.com/waggle-sensor/plugin-FCN-cloud"

CMD ["/usr/bin/python3", "plugin_node"]
