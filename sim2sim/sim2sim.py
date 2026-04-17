import time
import numpy as np

from collections import deque
from typing import Tuple, List, Literal


import os
import sys
import cv2

sys.path.append(os.path.dirname(__file__))

from mujoco_env import MujocoEnv
from config import *
from onnx_inference import OnnxPolicyInference as Inference


def quat_apply_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    # wxyz format quat apply inverse
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = np.cross(xyz, vec, axis=-1) * 2
    return (vec - quat[:, 0:1] * t + np.cross(xyz, t, axis=-1)).reshape(shape)


class RingBuffer:
    def __init__(self, shape: Tuple[int, int], length: int = 8):
        self.shape = shape
        self.length = length
        self.que = deque(maxlen=self.length)
        self.reset()

    def reset(self):
        self.que.clear()
        for _ in range(self.length):
            self.que.append(np.zeros(self.shape, dtype=np.float32))

    def append(self, data: np.ndarray):
        if self.shape != data.shape:
            raise ValueError(
                f"Data size mismatch expected {self.shape} while get {data.shape}"
            )
        self.que.append(data.copy())

    def get_history(self):
        return np.array(self.que).copy()


class DepthImagePipeline:
    def __init__(
        self,
        shape: Tuple[int, int] = (36, 64),
        length: int = 50,
        crop_region: Tuple[int, int, int, int] = tuple([0, 0, 0, 0]),
        near_far_clip: Tuple[float, float] = tuple([0, 2.5]),
        nb_frames: int = 8,
    ):
        self.shape = shape
        self.length = length
        self.que = deque(maxlen=length)
        self.crop_region = crop_region
        self.near_clip, self.far_clip = near_far_clip
        self.nb_frames = nb_frames

        down_sample_factor = int(self.length / 50 * 5)
        self.indices = np.linspace(
            -1 - down_sample_factor * (self.nb_frames - 1), -1, self.nb_frames
        ).astype(np.int32)
        self.reset()

    def reset(self):
        self.que.clear()
        h, w = self.shape
        y1, y2, x1, x2 = self.crop_region
        dh, dw = h - y1 - y2, w - x1 - x2
        for _ in range(self.length):
            self.que.append(np.zeros((dh, dw), dtype=np.float32))

    def show_image(self, image: np.ndarray):
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        cv2.imshow("current_depth_image", image)
        cv2.waitKey(1)

    def append(self, image: np.ndarray):
        # resize image
        h, w = self.shape
        image = cv2.resize(image, dsize=(w, h))

        # crop image
        y1, y2, x1, x2 = self.crop_region
        image = image[y1 : h - y2, x1 : w - x2]

        # inpainting image
        mask = (image < 0.2).astype(np.uint8)
        image = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)

        # gaussian blur
        image = cv2.GaussianBlur(image, (3, 3), 1, 1)

        # clip image
        np.clip(image, a_min=self.near_clip, a_max=self.far_clip, out=image)
        image = (image - self.near_clip) / (self.far_clip - self.near_clip)
        # self.show_image(image)
        self.que.append(image)

    def get_depth_obs(self):
        obs = []
        for index in self.indices:
            obs.append(self.que[index])
        obs = np.array(obs, dtype=np.float32)[None, ...]
        return obs


class Sim2simInstance:
    def __init__(self, sim2sim_type: Literal["stand", "parkour"]):
        self.sim2sim_type = sim2sim_type
        self.velocity_commands = self._default_velocity_commands()

        self.sim_env = MujocoEnv(
            mjcf_file=MJCF_FILE,
            sim_dt=SIM_DT,
            decimation=DECIMATION,
            depth_camera_name=DEPTH_CAMERA_NAME,
            depth_image_shape=DEPTH_IMAGE_SHAPE,
            use_secondray_imu=USE_SECONDARY_IMU,
            key_callback=self._key_callback,
        )

        policy_joint_name = POLICY_JOINT_NAMES
        robot_joint_name = UNITREE_G1_29DOF_SDK_JOINT_NAMES
        self.robot_to_policy = self.compute_joint_indices(
            robot_joint_name, policy_joint_name
        )
        self.policy_to_robot = self.compute_joint_indices(
            policy_joint_name, robot_joint_name
        )

        # initialize observation buffer
        self.obs_ring_buffer: List[RingBuffer] = []
        history_length = HISTORY_LENGTH
        obs_dims = OBS_DIMS
        for dim in obs_dims:
            self.obs_ring_buffer.append(RingBuffer(shape=(dim,), length=history_length))

        # depth image pipeline
        self.resized_depth_image_shape = RESIZED_DEPTH_IMAGE_SHAPE
        y1, y2, x1, x2 = OBS_DEPTH_IMAGE_CROP_REGION
        self.obs_depth_image_shape = (
            self.resized_depth_image_shape[0] - y1 - y2,
            self.resized_depth_image_shape[1] - x1 - x2,
        )
        self.depth_pipeline = DepthImagePipeline(
            shape=self.resized_depth_image_shape,
            crop_region=OBS_DEPTH_IMAGE_CROP_REGION,
        )

        # robot order
        self.action_scale = np.array(ACTION_SCALES, dtype=np.float32)[
            self.policy_to_robot
        ]
        self.default_joint_pos = np.array(DEFAULT_JOINT_POS, dtype=np.float32)[
            self.policy_to_robot
        ]
        self.joint_signs = np.array(JOINT_SIGNS, dtype=np.float32)[self.policy_to_robot]
        self.stiffness = np.array(STIFFNESS)[self.policy_to_robot]
        self.damping = np.array(DAMPING)[self.policy_to_robot]
        self.torque_limit = np.array(TORQUE_LIMITS)[self.policy_to_robot]

        # last action: policy order
        self.last_action = np.zeros((29,), dtype=np.float32)

        # policies
        if self.sim2sim_type == "stand":
            self.actor = Inference(STAND_POLICY_FILE)
            self.depth_encoder = Inference(STAND_DEPTH_ENCODER_FILE)
        else:
            self.actor = Inference(PARKOUR_POLICY_FILE)
            self.depth_encoder = Inference(PARKOUR_DEPTH_ENCODER_FILE)
            self._print_keyboard_help()

    def _default_velocity_commands(self) -> np.ndarray:
        return np.array((0.0, 0.0, 0.0), dtype=np.float32)

    def _print_keyboard_help(self):
        if self.sim2sim_type != "parkour":
            return
        print(
            "Keyboard controls enabled: "
            "[8] forward, [2] backward, [4] turn left, [6] turn right, [5] stop"
        )
        print(f"Initial velocity command: {self.velocity_commands.tolist()}")

    def _key_callback(self, keycode: int):
        if self.sim2sim_type != "parkour":
            return

        key_to_command = {
            KEY_KP_2: np.array((-0.20, 0.0, 0.0), dtype=np.float32),
            KEY_KP_4: np.array((0.0, 0.0, 0.45), dtype=np.float32),
            KEY_KP_5: np.array((0.0, 0.0, 0.0), dtype=np.float32),
            KEY_KP_6: np.array((0.0, 0.0, -0.45), dtype=np.float32),
            KEY_KP_8: np.array((0.6, 0.0, 0.0), dtype=np.float32),
        }
        command = key_to_command.get(keycode)
        if command is None:
            return
        self.velocity_commands = command
        print(
            "Updated velocity command "
            f"vx={self.velocity_commands[0]:.2f}, "
            f"vy={self.velocity_commands[1]:.2f}, "
            f"yaw={self.velocity_commands[2]:.2f}"
        )

    def update_observation(self):
        imu_offset = self.sim_env.imu_offset
        sensordata = self.sim_env.model_data.sensordata.astype(np.float32)
        # base angle velocity
        base_angle_vel = sensordata[imu_offset + 4 : imu_offset + 7] * 0.25
        self.obs_ring_buffer[0].append(base_angle_vel)

        # projected gravity
        root_quat = sensordata[imu_offset : imu_offset + 4]
        projected_gravity = quat_apply_inverse(
            root_quat, np.array([0, 0, -1], dtype=np.float32)
        )
        self.obs_ring_buffer[1].append(projected_gravity)

        # velocity commands
        if self.sim2sim_type == "stand":
            velocity_commands = np.array((0, 0.0, 0.0), dtype=np.float32)
        else:
            velocity_commands = self.velocity_commands
        self.obs_ring_buffer[2].append(velocity_commands)

        # joint pos rel
        joint_pos_rel = sensordata[:29] * self.joint_signs - self.default_joint_pos
        # policy order
        joint_pos_rel = joint_pos_rel[self.robot_to_policy]
        self.obs_ring_buffer[3].append(joint_pos_rel)

        # joint vel rel
        joint_vel_rel = sensordata[29:58] * self.joint_signs * 0.05
        joint_vel_rel = joint_vel_rel[self.robot_to_policy]  # policy order
        self.obs_ring_buffer[4].append(joint_vel_rel)

        # last action
        self.obs_ring_buffer[5].append(self.last_action)

        # depth image
        if self.sim2sim_type == "stand":
            image = np.zeros(self.obs_depth_image_shape, dtype=np.float32)
        else:
            image = self.sim_env.depth_render.render()
        self.depth_pipeline.append(image)

    def get_history_obs(self):
        history_obs = []
        # proprio observations
        for each in self.obs_ring_buffer:
            history_obs.append(each.get_history().ravel())

        # deptah image observation
        history_depth_image = history_depth_image = self.depth_pipeline.get_depth_obs()
        depth_image_feat = self.depth_encoder({"input": history_depth_image})
        history_obs.append(depth_image_feat.ravel())
        return np.concatenate(history_obs)

    def compute_joint_indices(
        self, joint_a: List[str], joint_b: List[str]
    ) -> np.ndarray:
        # find joint_b in joint_a
        if joint_b is None:
            return np.arange(len(joint_a), dtype=np.uint32)
        if joint_a is None:
            return np.arange(len(joint_b), dtype=np.uint32)
        indices = []
        for joint in joint_b:
            indices.append(joint_a.index(joint))
        return np.array(indices, dtype=np.uint32)

    def keep_running(self):
        decimation = DECIMATION
        eposide_length = 0
        while self.sim_env.viewer.is_running():
            time_start = time.perf_counter()
            # update
            self.update_observation()
            history_obs = self.get_history_obs()[None, ...]
            # policy order
            raw_action = self.actor({"input": history_obs}).ravel()
            self.last_action = raw_action.copy()

            # robot order
            action = (
                raw_action[self.policy_to_robot] * self.action_scale
                + self.default_joint_pos
            )
            action = action * self.joint_signs
            for _ in range(decimation):
                # pd control
                tau = (
                    self.stiffness * (action - self.sim_env.model_data.sensordata[:29])
                    - self.damping * self.sim_env.model_data.sensordata[29:58]
                )
                tau = np.clip(tau, a_min=-self.torque_limit, a_max=self.torque_limit)
                self.sim_env.model_data.ctrl[:] = tau
                self.sim_env.physical_step()
            self.sim_env.viewer.sync()
            eposide_length += 1

            time_to_sleep = SIM_DT * DECIMATION - (time.perf_counter() - time_start)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, choices=["parkour", "stand"], default="parkour"
    )
    args = parser.parse_args()

    instance = Sim2simInstance(args.task)
    instance.keep_running()
