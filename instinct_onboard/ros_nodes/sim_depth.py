import threading
import time
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from .idl._DepthImage_ import DepthImage_


class DDSDepthCamera:
    def __init__(
        self,
        resolution=(480, 270),
        topic="rt/raw_depth_image",
        domain_id=0,
        interface=None,
    ):
        self.resolution = resolution  # (width, height)
        self.topic = topic
        self.latest = np.zeros(resolution[::-1], dtype=np.float32)
        self.timestamp = 0.0
        self.has_data = False
        self.lock = threading.Lock()

        if interface is None:
            ChannelFactoryInitialize(domain_id)
        else:
            ChannelFactoryInitialize(domain_id, interface)

        self.sub = ChannelSubscriber(topic, DepthImage_)
        self.sub.Init(self._callback, 10)

    def _callback(self, msg: DepthImage_):
        height = int(msg.height)
        width = int(msg.width)
        if (width, height) != self.resolution:
            return

        arr = np.asarray(msg.data, dtype=np.float32)[: height * width].reshape(height, width)
        with self.lock:
            self.latest[:] = arr
            self.timestamp = time.time()
            self.has_data = True

    def get_camera_data(self):
        with self.lock:
            if not self.has_data:
                return None
            return self.latest.copy()
