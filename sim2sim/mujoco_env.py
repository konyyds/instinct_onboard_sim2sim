import os
import mujoco
import mujoco.viewer
import numpy as np

from typing import Tuple


class DepthImageRender:
    def __init__(
        self,
        camera_name: str,
        model: mujoco.MjModel,
        model_data: mujoco.MjData,
        height: int,
        width: int,
    ):
        self._camera_name = camera_name
        self._model = model
        self._model_data = model_data
        self._render = mujoco.Renderer(model, height, width)
        self._render.enable_depth_rendering()
        self._data = np.zeros((height, width), dtype=np.float32)

    def render(self, model_data_lock=None) -> np.ndarray:
        if model_data_lock is not None:
            with model_data_lock:
                self._render.update_scene(self._model_data, self._camera_name)
        else:
            self._render.update_scene(self._model_data, self._camera_name)
        self._render.render(out=self._data)
        return self._data


class MujocoEnv:
    def __init__(
        self,
        mjcf_file: str,
        sim_dt: float = 0.005,
        decimation: int = 4,
        depth_camera_name: str = "depth_camera",  # check in g1_29dof.xml
        depth_image_shape: Tuple[int, int] = (270, 480),  # height, width
        use_secondray_imu: bool = True,
        key_callback=None,
    ):
        self.mjcf_file = mjcf_file
        self.sim_dt = sim_dt
        self.decimation = decimation

        self.depth_camera_name = depth_camera_name
        self.depth_image_shape = depth_image_shape

        self.use_secondary_imu = use_secondray_imu
        self.key_callback = key_callback

        self._init_model()
        self._init_viewer()
        self._init_depth_render()

    def _init_model(self):
        if not self.mjcf_file.endswith(".xml"):
            raise ValueError(f"Unsupport file type, MujocoEnv only support xml file.")
        if not os.path.exists(self.mjcf_file):
            raise FileNotFoundError(f"No such file called {self.mjcf_file}")
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_file)
        self.model_data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.sim_dt

        self.nb_actuators = self.model.nu
        self.nb_sensors = self.model.nsensor
        self.imu_offset = self.nb_actuators * 3  # (pos, vel, torque) of each actuator
        next_sensor_name = mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, self.imu_offset
        )
        self.have_imu = True if next_sensor_name == "imu_quat" else False
        if self.use_secondary_imu:
            secondary_imu_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, "secondary_imu_quat"
            )
            self.imu_offset = self.model.sensor_adr[secondary_imu_id]

    def _init_viewer(self):
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.model_data, key_callback=self.key_callback
        )

    def _init_depth_render(self):
        self.depth_render = DepthImageRender(
            self.depth_camera_name,
            self.model,
            self.model_data,
            self.depth_image_shape[0],
            self.depth_image_shape[1],
        )

    def render_depth_image(self):
        return self.depth_render.render()

    def forward(self):
        mujoco.mj_forward(self.model, self.model_data)

    def physical_step(self):
        mujoco.mj_step(self.model, self.model_data)
