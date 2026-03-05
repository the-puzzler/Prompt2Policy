"""Lightweight MuJoCo GLFW viewer that shows the scene from a fixed camera POV."""

import glfw
import mujoco
import numpy as np
from OpenGL.GL import glReadPixels, GL_RGB, GL_UNSIGNED_BYTE


class MujocoViewer:
    """Opens a GLFW window rendering the MuJoCo scene from a named camera."""

    def __init__(self, model, data, camera_id, width=640, height=480, title="MuJoCo Viewer"):
        self.model = model
        self.data = data
        self.camera_id = camera_id
        self.width = width
        self.height = height

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)

        # MuJoCo rendering context
        self.scene = mujoco.MjvScene(model, maxgeom=1000)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.viewport = mujoco.MjrRect(0, 0, width, height)

        # Camera set to the fixed camera
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = camera_id

        self.opt = mujoco.MjvOption()

    def is_alive(self):
        return not glfw.window_should_close(self.window)

    def sync(self):
        if not self.is_alive():
            return

        glfw.make_context_current(self.window)

        # Update viewport in case window was resized
        w, h = glfw.get_framebuffer_size(self.window)
        self.viewport.width = w
        self.viewport.height = h

        # Update scene and render
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(self.viewport, self.scene, self.context)

        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        if self.window:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()
