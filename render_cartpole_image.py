#!/usr/bin/env python3
"""Render a single image of the visual cartpole URDF."""

import os
import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

CARTPOLE_VISUAL_SCALE = 4.5

visual_urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'safe_control_gym', 'envs', 'gym_control',
                           'assets', 'cartpole_visual.urdf')

client = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
p.resetSimulation(physicsClientId=client)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=client)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)

cartpole_id = p.loadURDF(visual_urdf, [0, 0, 0],
                          globalScaling=CARTPOLE_VISUAL_SCALE,
                          physicsClientId=client)

# Set pole at an angle, cart at x=-5.5 (like start state)
p.resetJointState(cartpole_id, jointIndex=0,
                  targetValue=-5.5 / CARTPOLE_VISUAL_SCALE,
                  physicsClientId=client)
p.resetJointState(cartpole_id, jointIndex=1,
                  targetValue=1.358, physicsClientId=client)

cam_view = p.computeViewMatrixFromYawPitchRoll(
    cameraTargetPosition=[0, 0, 0.3],
    distance=7.0,
    yaw=0,
    pitch=-5,
    roll=0,
    upAxisIndex=2,
    physicsClientId=client
)
cam_pro = p.computeProjectionMatrixFOV(
    fov=60,
    aspect=1920.0 / 1080,
    nearVal=0.01,
    farVal=1000.0
)

(w, h, rgb, _, _) = p.getCameraImage(
    width=1920, height=1080,
    shadow=1,
    renderer=p.ER_TINY_RENDERER,
    viewMatrix=cam_view, projectionMatrix=cam_pro,
    physicsClientId=client
)
p.disconnect(client)

img = np.reshape(rgb, (h, w, 4))[:, :, :3]
Image.fromarray(img).save('cartpole_preview.png')
print("Saved cartpole_preview.png")
