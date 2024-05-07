import math
import time
import queue

import cv2
import carla
import random

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import os
import pathlib

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def display_img(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)

def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name,
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir))

  return model

PATH_TO_LABELS = 'C:/Users/User/Documents/TensorFlow/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def show_inference(model, image):
    image_np = image
    output_dict = run_inference_for_single_image(model, image_np)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)
    display(Image.fromarray(image_np))
    return image_np

def change_weather(world):
    clear_weather = carla.WeatherParameters.ClearNoon
    world.set_weather(clear_weather)

    custom_weather = carla.WeatherParameters(
        cloudiness=20.0,
        precipitation=130.0,
        sun_altitude_angle=80.0)
    world.set_weather(custom_weather)

def image_process(image):
    # image.save_to_disk('out/%06d.png' % image.frame)
    img = np.array(image.raw_data)
    img = img.reshape((600, 800, 4))
    img = img[:, :, :3]

    img = show_inference(detection_model, img)

    cv2.imshow('image', img)
    cv2.waitKey(1)
    pass


def process_lidar_data(data):
    points = data.raw_data
    num_points = len(points) // 3

    print(f"Received {num_points} points in this frame.")
    if num_points > 0:
        distances = [math.sqrt(x ** 2 + y ** 2 + z ** 2) for x, y, z in zip(points[0::3], points[1::3], points[2::3])]
        average_distance = sum(distances) / num_points
        print(f"Average distance of points from sensor: {average_distance:.2f} meters")

def visualize_point_cloud(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[0::3], points[1::3], points[2::3], s=1)
    plt.show()


def get_speed(vehicle):
    vel = vehicle.get_velocity()
    return 3.6*math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

class VehiclePIDController():
    def __init__(self, vehicle, args_longitudinal, args_lateral, max_throttle=0.75, max_brake=0.3, max_steering=0.8):
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.past_steering = self.vehicle.get_control().steer
        self.max_throttle = max_throttle
        self.max_brake = max_brake
        self.max_steering = max_steering

        self.long_controller = PIDLongitudinalController(self.vehicle, **args_longitudinal)
        self.lat_controller = PIDLateralController(self.vehicle, **args_lateral)

    def run_step(self, target_speed, waypoint):
        acceleration = self.long_controller.run_step(target_speed)
        current_steering = self.lat_controller.run_step(waypoint)
        control = carla.VehicleControl()

        if acceleration >= 0.0:
            control.throttle = min(abs(acceleration), self.max_throttle)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        if current_steering > self.past_steering+0.1:
            current_steering = self.past_steering+0.1
        elif current_steering < self.past_steering-0.1:
            current_steering = self.past_steering-0.1

        if current_steering >= 0:
            steering = min(self.max_steering, abs(current_steering))
        else:
            steering = max(-self.max_steering, abs(current_steering))

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering

        return control

class PIDLongitudinalController():
    def __init__(self, vehicle, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        self.vehicle = vehicle
        self.K_P = K_P
        self.K_I = K_I
        self.K_D = K_D
        self.dt = dt
        self.errorBuffer = queue.deque(maxlen=10)

    def pid_controller(self, target_speed, current_speed):
        error = target_speed - current_speed
        self.errorBuffer.append(error)

        if len(self.errorBuffer) >= 2:
            ie = sum(self.errorBuffer) * self.dt
            de = (self.errorBuffer[-1] - self.errorBuffer[-2]) / self.dt

        else:
            ie = 0.0
            de = 0.0

        return np.clip(self.K_P * error + self.K_I * ie + self.K_D * de, -1.0, 1.0)

    def run_step(self, target_speed):
        current_speed = get_speed(self.vehicle)
        return self.pid_controller(target_speed, current_speed)


class PIDLateralController():
    def __init__(self, vehicle, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        self.vehicle = vehicle
        self.K_P = K_P
        self.K_I = K_I
        self.K_D = K_D
        self.dt = dt
        self.errorBuffer = queue.deque(maxlen=10)

    def run_step(self, waypoint):
        return self.pid_controller(waypoint, self.vehicle.get_transform())

    def pid_controller(self, waypoint, vehicle_transform):
        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                         y=math.sin(math.radians(vehicle_transform.rotation.yaw)))
        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])

        w_vec = np.array([waypoint.transform.location.x - v_begin.x,
                          waypoint.transform.location.y - v_begin.y,
                          0.0])
        dot = math.acos(np.clip(np.dot(w_vec, v_vec) / np.linalg.norm(w_vec) * np.linalg.norm(v_vec), -1.0, 1.0))
        cross = np.cross(v_vec, w_vec)
        if cross[2]<0:
            dot *= -1

        self.errorBuffer.append(dot)

        if len(self.errorBuffer) > 2:
            ie = sum(self.errorBuffer) * self.dt
            de = (self.errorBuffer[-1] - self.errorBuffer[-2]) / self.dt

        else:
            ie = 0.0
            de = 0.0

        return np.clip((self.K_P*dot) + (self.K_I*ie) + (self.K_D*de), -1.0, 1.0)

def main():
    actor_list = []
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        world = client.load_world('Town04')
        map = world.get_map()

        change_weather(world)

        bp = world.get_blueprint_library()

        vehicle_bp = bp.filter('cybertruck')[0]
        # spawnpoint = carla.Transform(carla.Location(x=130, y=195, z=15),
        #                             carla.Rotation(yaw=0, roll=0, pitch=0))
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))
        # # Set the base path
        # base_path = "C:/Users/User/carla/Build/chrono-install/share/chrono/data/vehicle/"
        #
        # # Set the template files
        #
        # vehicle_json = "sedan/vehicle/Sedan_Vehicle.json"
        # powertrain_json = "sedan/powertrain/Sedan_SimpleMapPowertrain.json"
        # tire_json = "sedan/tire/Sedan_TMeasyTire.json"
        #
        # # Enable Chrono physics
        #
        # vehicle.enable_chrono_physics(5000, 0.002, vehicle_json, powertrain_json, tire_json, base_path)
        actor_list.append(vehicle)
        #
        # rgb_bp = bp.find('sensor.camera.rgb')
        # rgb_bp.set_attribute('image_size_x', '800')
        # rgb_bp.set_attribute('image_size_y', '600')
        # rgb_bp.set_attribute('fov', '90')
        # rgb_bp.set_attribute('sensor_tick', '0.1')
        # rgb_tf = carla.Transform(carla.Location(x=1.5, z=2.4))
        # rgb = world.spawn_actor(rgb_bp,
        #                         rgb_tf,
        #                         attach_to=vehicle)
        # actor_list.append(rgb)
        # rgb.listen(lambda image: image_process(image))

        # dep_bp = bp.find('sensor.camera.depth')
        # dep_bp.set_attribute('image_size_x', '800')
        # dep_bp.set_attribute('image_size_y', '600')
        # dep_bp.set_attribute('fov', '90')
        # dep_bp.set_attribute('sensor_tick', '0.1')
        # dep_tf = carla.Transform(carla.Location(x=1.5, z=2.4))
        # dep = world.spawn_actor(dep_bp,
        #                         dep_tf,
        #                         attach_to=vehicle)
        # actor_list.append(dep)
        # dep.listen(lambda image: image_process(image))

        lidar_bp = bp.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('points_per_second', '56000')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('range', '100')
        lidar_bp.set_attribute('upper_fov', '10')
        lidar_bp.set_attribute('lower_fov', '-30')
        lidar_bp.set_attribute('sensor_tick', '0.1')
        lidar_tf = carla.Transform(carla.Location(x=1.5, z=2.4))
        lidar = world.spawn_actor(lidar_bp, lidar_tf, attach_to=vehicle)
        actor_list.append(lidar)
        lidar.listen(lambda cloud_point: process_lidar_data(cloud_point))

        control_vehicle = VehiclePIDController(vehicle,
                                               args_longitudinal={'K_P': 1, 'K_I': 0.0, 'K_D': 0.0},
                                               args_lateral={'K_P': 1, 'K_I': 0.0, 'K_D': 0.0})

        while True:
            waypoints = world.get_map().get_waypoint(vehicle.get_location())
            waypoint = np.random.choice(waypoints.next(0.3))
            control_signal = control_vehicle.run_step(5, waypoint)
            vehicle.apply_control(control_signal)

        time.sleep(15)

    finally:
        for actor in actor_list:
            actor.destroy()
 
        print('end')

if __name__ == '__main__':
    main()
