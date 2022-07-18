import bpy
import bpy_extras
import os
import random
import numpy as np
import mathutils
import time
import math
import csv
import copy
from warnings import warn

########################################################################
user = 1
# 0: Paul
# 1: Filip

user0_path = r"C:\Users\Paul\PycharmProjects\LegoClassification\\"
user1_path = r"C:\Users\sokfi\Code\jupyter\LegoClassification\\"
########################################################################

ROOT = user0_path if not user else user1_path
LDRAW_PATH = ROOT + r"dataset\ldraw\\"
OUT_PATH = ROOT + r"data\renders_2\\"
OUT_CSV = ROOT + r"data\RenderedParts_2.csv"


# TODO: Crop around the actual part, not its bounding box
# TODO: Sort the 1k most common parts by similarity, e.g part 92589 and 60621 should be the same class
# TODO: The crop should always be a square, do not simply retry if it isn't


class CamSettings:
    def __init__(self, pos_y, height, rot, horizontal_fov, sensor_height, sensor_width, height_range=0., rot_range=0.):
        """
        :param pos_y:
        :param height: cam z position
        :param rot: cam rotation in degrees. 0 = cam looks straight down at the ground
        :param horizontal_fov:
        :param sensor_height:
        :param sensor_width:
        :param height_range: optionally uniformly randomize cam z position within height+-range
        :param rot_range: optionally uniformly randomize cam rotation within rot+-range
        """
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.horizontal_fov = horizontal_fov
        self.cam_height_range = height_range
        self.cam_angle_range = rot_range
        self.rot = rot
        self.pos_y = pos_y
        self.height = height

    def apply(self):
        """Apply settings to the blender camera object"""
        cam = self.cam()
        cam.location.x = 0
        cam.location.y = self.pos_y
        cam.location.z = self.height + random.uniform(-self.cam_height_range, self.cam_height_range)
        cam.rotation_euler[0] = math.radians(self.rot + random.uniform(-self.cam_angle_range, self.cam_angle_range))
        cam.rotation_euler[1] = 0
        cam.rotation_euler[2] = 0
        cam.data.sensor_fit = 'VERTICAL'
        cam.data.sensor_width = self.sensor_width
        cam.data.sensor_height = self.sensor_height
        cam.data.angle = math.radians(self.horizontal_fov)

    @staticmethod
    def cam():
        """Get the blender camera object"""
        return bpy.data.objects["Camera"]


class RaspiCamV1(CamSettings):
    """Settings to emulate the Raspberry Camera V1"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs, horizontal_fov=53.5, sensor_height=2.74, sensor_width=3.76)


class RenderSettings:
    def __init__(self, device='GPU', crop_mode='part', random_crop_pad_std=0.001, samples=64, max_bounces=4,
                 diffuse_bounces=3, glossy_bounces=3,
                 transmission_bounces=3,
                 transparent_max_bounces=3, use_persistent_data=True, resolution_x=720, resolution_y=720,
                 tile_size=720):
        """
        :param device:
        :param crop_mode: all: render entire camera view. boundbox: crop to the part 3D bound box. part: crop to part as seen in 2D cam view
        :param random_crop_pad_std: Add a random padding to the crop
        :param samples:
        :param max_bounces:
        :param diffuse_bounces:
        :param glossy_bounces:
        :param transmission_bounces:
        :param transparent_max_bounces:
        :param use_persistent_data:
        :param resolution_x:
        :param resolution_y:
        :param tile_size:
        """
        crop_modes = ['all', 'boundbox', 'part']
        if crop_mode not in crop_modes:
            raise ValueError(f"Invalid crop mode, allowed: {crop_modes}")

        self.crop_mode = crop_mode
        self.random_crop_pad_std = random_crop_pad_std
        self.device = device
        self.samples = samples
        self.max_bounces = max_bounces
        self.diffuse_bounces = diffuse_bounces
        self.glossy_bounces = glossy_bounces
        self.transmission_bounces = transmission_bounces
        self.transparent_max_bounces = transparent_max_bounces
        self.use_persistent_data = use_persistent_data
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.tile_size = tile_size

        bpy.context.scene.render.border_min_x = 0.0
        bpy.context.scene.render.border_min_y = 0.0
        bpy.context.scene.render.border_max_x = 1.0
        bpy.context.scene.render.border_max_y = 1.0

    def get_crop_quot(self):
        """Get the quotient of the sides of the current image crop. 1 means the crop is square."""
        min_x = bpy.context.scene.render.border_min_x
        min_y = bpy.context.scene.render.border_min_y
        max_x = bpy.context.scene.render.border_max_x
        max_y = bpy.context.scene.render.border_max_y

        if min_x == max_x or min_y == max_y:
            return 0

        ratio = float(self.resolution_x) / self.resolution_y
        return ratio * (max_x - min_x) / (max_y - min_y)

    def apply_crop(self, part):
        """Crop the image to be rendered with the crop_mode specified in init"""
        bpy.data.scenes[0].render.use_border = True
        bpy.data.scenes[0].render.use_crop_to_border = True

        scene = bpy.context.scene
        cam = bpy.data.objects["Camera"]
        bpy.context.view_layer.update()

        min_x = 2.0
        min_y = 2.0
        max_x = -2.0
        max_y = -2.0

        if self.crop_mode == 'boundbox':
            # for each vertex of the bound box, get its screen coords and keep track of the min & max x and y
            for pt in part.bound_box:
                real_pos = part.matrix_world @ mathutils.Vector(pt)
                scr_pos = bpy_extras.object_utils.world_to_camera_view(scene, cam, real_pos)
                if scr_pos[0] < min_x:
                    min_x = scr_pos[0]
                elif scr_pos[0] > max_x:
                    max_x = scr_pos[0]
                if scr_pos[1] < min_y:
                    min_y = scr_pos[1]
                elif scr_pos[1] > max_y:
                    max_y = scr_pos[1]
        elif self.crop_mode == 'part':
            # for each vertex of the part, get its screen coords and keep track of the min & max x and y
            for v in part.data.vertices:
                real_pos = part.matrix_world @ mathutils.Vector(v.co)
                scr_pos = bpy_extras.object_utils.world_to_camera_view(scene, cam, real_pos)
                if scr_pos[0] < min_x:
                    min_x = scr_pos[0]
                elif scr_pos[0] > max_x:
                    max_x = scr_pos[0]
                if scr_pos[1] < min_y:
                    min_y = scr_pos[1]
                elif scr_pos[1] > max_y:
                    max_y = scr_pos[1]

        # make the crop square todo: fix the float rounding errors and make the crop truly square
        ratio = float(self.resolution_x) / self.resolution_y
        quot = ratio * (max_x - min_x) / (max_y - min_y)
        if quot > 1:
            diff = ratio * (max_x - min_x) - (max_y - min_y)
            max_y += diff / 2
            min_y -= diff / 2
        elif quot < 1:
            diff = -(max_x - min_x) + (max_y - min_y) / ratio
            max_x += diff / 2
            min_x -= diff / 2

        # random pad
        rand_tl = abs(np.random.normal(loc=0, scale=self.random_crop_pad_std))
        rand_tr = abs(np.random.normal(loc=0, scale=self.random_crop_pad_std))
        rand_br = abs(np.random.normal(loc=0, scale=self.random_crop_pad_std))
        rand_bl = abs(np.random.normal(loc=0, scale=self.random_crop_pad_std))

        # check whether the pad is out of screen bounds and adjust
        if min_x - (rand_tl + rand_bl) < 0:
            rand_tl = rand_bl = 0
        if min_y - (rand_br + rand_bl) * ratio < 0:
            rand_br = rand_bl = 0
        if max_x + (rand_tr + rand_br) > 1:
            rand_tr = rand_br = 0
        if max_y + (rand_tl + rand_tr) * ratio > 1:
            rand_tl = rand_tr = 0

        # set the crop to the min & max x and y with a random pad
        bpy.context.scene.render.border_min_x = min_x - (rand_tl + rand_bl)
        bpy.context.scene.render.border_min_y = min_y - (rand_br + rand_bl) * ratio
        bpy.context.scene.render.border_max_x = max_x + (rand_tr + rand_br)
        bpy.context.scene.render.border_max_y = max_y + (rand_tl + rand_tr) * ratio

        return self.get_crop_quot()

    def apply(self, part=None):
        """Apply the render settings and crop depending on crop_mode"""
        bpy.data.scenes[0].cycles.device = self.device
        bpy.data.scenes[0].cycles.samples = self.samples
        bpy.data.scenes[0].cycles.max_bounces = self.max_bounces
        bpy.data.scenes[0].cycles.diffuse_bounces = self.diffuse_bounces
        bpy.data.scenes[0].cycles.glossy_bounces = self.glossy_bounces
        bpy.data.scenes[0].cycles.transmission_bounces = self.transmission_bounces
        bpy.data.scenes[0].cycles.transparent_max_bounces = self.transparent_max_bounces
        bpy.data.scenes[0].render.use_persistent_data = self.use_persistent_data
        bpy.data.scenes[0].render.resolution_x = self.resolution_x
        bpy.data.scenes[0].render.resolution_y = self.resolution_y
        bpy.data.scenes[0].cycles.tile_size = self.tile_size

        if self.crop_mode == 'all':
            bpy.context.scene.render.border_min_x = 0
            bpy.context.scene.render.border_min_y = 0
            bpy.context.scene.render.border_max_x = 1
            bpy.context.scene.render.border_max_y = 1
        else:
            if part is None:
                bpy.context.scene.render.border_min_x = 0
                bpy.context.scene.render.border_min_y = 0
                bpy.context.scene.render.border_max_x = 1
                bpy.context.scene.render.border_max_y = 1
                return False
            self.apply_crop(part)


def clear_parts():
    """Delete all objects which contain .dat in their name (all ldraw lego parts)"""
    bpy.ops.object.select_all(action='DESELECT')
    for i in bpy.data.objects.keys():
        if '.dat' in i:
            bpy.context.view_layer.objects.active = bpy.data.objects[i]
            bpy.data.objects[i].select_set(True)
            bpy.ops.object.delete()


def get_new_part(name=None, delete_plate=False, exclude_list=None):
    """
    Load new LDraw part into blender with a name or random

    :param exclude_list: Optionally exclude parts with names
    :param delete_plate: Whether to delete the default ldraw lego ground plate when importing
    :param name: The name of the ldraw part. If none, get a random one.
    """
    get_rand = False
    if not name:
        part_file = random.choice(os.listdir(LDRAW_PATH + "\\parts\\"))
        get_rand = True
    elif not os.path.exists(LDRAW_PATH + "\\parts\\" + name):
        return None
    else:
        part_file = name

    if exclude_list is not None and part_file in exclude_list:
        if get_rand:
            while part_file in exclude_list:
                part_file = random.choice(os.listdir(LDRAW_PATH + "\\parts\\"))
        else:
            return None

    bpy.ops.import_scene.importldraw(filepath=LDRAW_PATH + "\\parts\\" + part_file, ldrawPath=LDRAW_PATH,
                                     importCameras=False, numberNodes=False, positionCamera=False, resPrims='High')

    part = None
    for i in bpy.data.objects.keys():
        if '.dat' in i:
            part = bpy.data.objects[i]
            break

    if delete_plate:
        bpy.ops.object.select_all(action='DESELECT')
        for i in bpy.data.objects.keys():
            if 'LegoGroundPlane' in i:
                bpy.data.objects[i].select_set(True)
                bpy.ops.object.delete()
                break

    return part


def rand_rotation(part, deg_of_freedom='XYZ'):
    """Rotate a blender object randomly in the specified axes

    :param part:
    :param deg_of_freedom: The axes around which the part should be randomly rotated. E.g. 'X': only rotate around x-axis. 'XY': rotate around both x and y
    """
    if not deg_of_freedom:
        return
    if 'x' in deg_of_freedom.lower():
        part.rotation_euler[0] = random.uniform(0, 2 * math.pi)
    if 'y' in deg_of_freedom.lower():
        part.rotation_euler[1] = random.uniform(0, 2 * math.pi)
    if 'z' in deg_of_freedom.lower():
        part.rotation_euler[2] = random.uniform(0, 2 * math.pi)


def seconds_to_time(seconds):
    """
    Return Hrs:Min.Sec from seconds
    :param seconds:
    """
    hrs = (seconds - (seconds % 3600)) / 3600
    seconds -= hrs * 3600
    min = (seconds - (seconds % 60)) / 60
    seconds -= min * 60
    return "%02d:%02d:%.4g" % (int(hrs), int(min), seconds)


def calc_mass(part):
    """Calculate mass of a part given its volume and average lego part density"""
    density = 0.095

    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = part
    part.select_set(True)

    polygons = bpy.context.active_object.data.polygons
    vertices = bpy.context.active_object.data.vertices

    volume = 0
    for polygon in polygons:
        v1, v2, v3 = vertices[polygon.vertices[0]].co, vertices[polygon.vertices[1]].co, vertices[polygon.vertices[2]].co
        volume += v1.dot(v2.cross(v3)) / 6.0

    return density * abs(volume)


# Todo Split this into more subfunctions for readability
def generate_dataset(render_settings, room_path, part_csv, colors, iters_per_part, pics_per_iter, cam_settings,
                     amount=999999, exclude_list=None,
                     simulate_physics=True, conveyor_speed=0.2, start_pos_z=0., start_pos_y_std=0., start_pos_x_std=0.1,
                     min_dist_to_cam=1.0, overwrite_existing=True, light_energy=1000, max_sim_steps=320, max_retry_count=8, drop_height_range=(-0.5, 0.5)):
    """
    Generate a dataset of rendered images

    :param drop_height_range: Part will be dropped from height +- range
    :param max_retry_count: Maximum amount of attempts to render a part before it is skipped (a non-square image crop is a failed attempt)
    :param max_sim_steps: Max amount of physics sim steps when dropping the part before locking it in place, in case the part hasn't stopped moving by then.
    :param render_settings: A RenderSettings object containing Blender render settings
    :param room_path: Path to the conveyor room path
    :param part_csv: Path to the csv file of part file names
    :param colors: A list of colors to chose from when rendering a part, should be common lego colors
    :param iters_per_part: The amount of times each specific part is put through the conveyor (iters)
    :param pics_per_iter: The amount of pictures to be taken each time a part passes through the conveyor room
    :param cam_settings: A CamSettings object containing Blender camera settings
    :param amount: Only render images of the first X parts from the part_csv file
    :param exclude_list: Part names to be skipped
    :param simulate_physics: Drop a part from some height and simulate physics each iteration for realistic positions
    :param conveyor_speed:
    :param start_pos_z: The height from which the part is to be dropped
    :param start_pos_y_std: Standard deviation of a part's y coordinate from 0
    :param start_pos_x_std: Standard deviation of a part's x coordinate from 0
    :param min_dist_to_cam: Part positions will be interpolated between part y start position and cam.y - min_dist_to_cam
    :param overwrite_existing: If an image with the same part name and iteration already exists, skip render
    :param light_energy: Blender light energy of the conveyor room lights
    """
    t1 = time.time()

    # clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # open the conveyor room
    bpy.ops.import_scene.fbx(filepath=room_path)
    bpy.context.view_layer.update()

    color_range = (-0.003, 0.003)  # range to slightly randomize lego colors
    light_col_range = (-0.03, 0.)  # slightly randomize light color
    light_pos_range = (-1, 1)  # slightly randomize light positions
    light_energy_range = (-25, 25)  # slightly randomize light energy

    # select all lights and add randomization
    cam = bpy.data.objects["Camera"]
    bpy.context.scene.camera = cam
    plane = bpy.data.objects["Plane"]
    lights = [bpy.data.objects[i] for i in bpy.data.objects.keys() if 'Light' in i]
    light_def_vals = {}

    for light in lights:
        light.data.energy = light_energy
        light_def_vals[light.name] = {}
        light_def_vals[light.name]['color'] = copy.deepcopy(light.data.color)
        light_def_vals[light.name]['energy'] = copy.deepcopy(light.data.energy)
        light_def_vals[light.name]['location'] = copy.deepcopy(light.location)

    # rot_x = cam.rotation_euler[0]
    # direction = mathutils.Vector((0, math.sin(rot_x), -math.cos(rot_x)))
    # hit, loc, norm, idx, ob, M = bpy.context.scene.ray_cast(bpy.context.evaluated_depsgraph_get(), cam.location, direction, distance=100000)
    # cam_mid = loc

    # render empty conveyor belt
    clear_parts()
    cam_settings.apply()
    render_settings.apply()
    bpy.data.scenes[0].render.filepath = os.path.abspath(os.path.join(OUT_PATH, os.pardir)) + '\\empty.png'
    # bpy.ops.render.render(write_still=True)

    header = None
    rendered_parts = []
    not_found_parts = []
    unrenderable_parts = []

    with open(part_csv, 'r') as file:
        reader = csv.reader(file)

        # for each part in parts_csv file
        for i, row in enumerate(reader):
            if i == 0:
                header = row
                continue
            elif amount <= 0:  # break if required amount of parts has been rendered
                break

            to_render = [x for x in range(iters_per_part)]  # array holding the indeces of the iterations that still need to be rendered

            name = row[2]
            if not overwrite_existing:  # Skip existing images of parts if overwrite_existing is False

                def iter_done(idx):
                    for j in range(pics_per_iter):
                        if not os.path.exists(OUT_PATH + name + '_' + str(idx) + '_' + str(j) + '.png'):
                            return False
                    return True

                to_render = [x for x in range(iters_per_part) if not iter_done(x)]  # render all iters that have less than pics_per_iter images
                if not to_render:
                    rendered_parts.append(row)  # part is considered done and skipped if all iter indeces found in images
                    continue

            clear_parts()

            # Attempt to get the next ldraw part
            try:
                part = get_new_part(name=name + '.dat', exclude_list=exclude_list, delete_plate=True)
                if part is None:
                    raise RuntimeError
            except:
                not_found_parts.append(name + '.dat')
                print('--- CAN\'T FIND PART, SKIPPING: ' + name + '.dat ---')
                continue

            rendered_parts.append(row)

            def_col = part.data.materials[0].node_tree.nodes["Group"].inputs[0].default_value[:]  # Save the part's default color

            bpy.context.scene.frame_end = max_sim_steps + pics_per_iter  # Set the amount of animation scenes to accomodate for the physics sim

            part_mass = calc_mass(part)  # Get the part's mass

            # Configure the default (non-randomized) position
            part.location[2] += start_pos_z
            max_pos_y = (abs(cam_settings.pos_y) - abs(min_dist_to_cam)) * np.sign(cam_settings.pos_y)
            start_pos = part.location.copy()
            start_rot = part.rotation_euler.copy()

            # For each iteration (the amount of times a part goes through the conveyor room, set by iters_per_part)
            for x in range(len(to_render)):
                # reset scene and part position
                bpy.ops.object.select_all(action='DESELECT')
                bpy.context.scene.frame_set(0)

                part.location = start_pos
                part.rotation_euler = start_rot

                cam_settings.apply()

                # randomize start rotation
                rand_rotation(part, 'XYZ')

                # pick a color
                new_col = random.choice(colors)
                new_col = [max(min(i / 255 + random.uniform(*color_range), 0.96), 0.04) for i in new_col]
                new_col.append(1)
                part.data.materials[0].node_tree.nodes["Group"].inputs[0].default_value = new_col

                # randomize lighting
                for light in lights:
                    light.data.color[2] += random.uniform(*light_col_range)
                    light.data.energy += random.uniform(*light_energy_range)
                    light.location[:2] = [light.location[i] + random.uniform(*light_pos_range) for i in range(2)]

                if simulate_physics:
                    part.location[2] += random.uniform(*drop_height_range)

                    # set up physics rigidbodies
                    bpy.ops.object.select_all(action='DESELECT')

                    # plane rigidbody
                    plane.select_set(True)
                    bpy.context.view_layer.objects.active = plane
                    bpy.ops.rigidbody.objects_add(type='PASSIVE')
                    plane.rigid_body.friction = 0.75
                    plane.select_set(False)

                    # part rigidbody
                    part.select_set(True)
                    bpy.context.view_layer.objects.active = part
                    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
                    try:
                        bpy.ops.rigidbody.objects_add(type='ACTIVE')
                    except Exception as ex:  # Sometimes adding rigidbody doesn't work for whatever reason
                        warn(f'--- CAN\'T ADD RIGIDBODY, SKIPPING PART {part.name} ({str(ex)}) ---')
                        rendered_parts.remove(row)
                        row.append('Reason: Unable to add rigidbody')
                        unrenderable_parts.append(row)
                        break

                    part.rigid_body.mass = part_mass  # Set the part mass to the one calculated

                    # step through the simulation until the part stops moving or max_sim_steps is reached
                    prev_mat = None
                    for step in range(max_sim_steps):
                        if prev_mat == part.matrix_world and step > 3:
                            break
                        prev_mat = part.matrix_world.copy()
                        bpy.context.scene.frame_set(step)

                    # lock the part and remove the rigidbody
                    bpy.ops.object.visual_transform_apply()
                    bpy.ops.rigidbody.object_remove()
                    part.select_set(False)

                # reset part x and y pos
                part.location[0] = start_pos[0]
                part.location[1] = start_pos[1]

                # randomize x and y pos
                part.location[0] += np.random.normal(0, start_pos_x_std)
                part.location[1] += np.random.normal(0, start_pos_y_std)

                valid_locations = []

                # Move the part on the conveyor belt and generate pics_per_iter amount of images
                if conveyor_speed > 0:
                    if pics_per_iter == 1:
                        part.location[1] = max_pos_y
                    else:  # TODO: Garbage fucking code
                        # check if the image is square, reposition if not
                        quot = render_settings.apply_crop(part)
                        if not math.isclose(quot, 1.0, rel_tol=1e-2):
                            # Repeatedly reposition until it's square.
                            for count in range(1, max_retry_count + 1):
                                print(f'Adjusting piece position {part.name}')
                                part.location[0] = start_pos[0] + np.random.normal(0, start_pos_x_std / count)
                                part.location[1] = start_pos[1] + np.random.normal(0, start_pos_y_std / count)
                                quot = render_settings.apply_crop(part)
                                count += 1
                                if math.isclose(quot, 1.0, rel_tol=1e-2):  # success
                                    break

                            # If max_retry_count has been reached, skip the part entirely. Todo: This should never happen
                            if count == max_retry_count:
                                warn(f'--- BAD IMAGE DIMENSIONS, SKIPPING PART {part.name} ---')
                                row.append('Reason: Bad image dimensions')
                                rendered_parts.remove(row)
                                unrenderable_parts.append(row)
                                break

                        # While the crop is square, move the part slightly forward and recheck if square. If yes, add pos to list of valid positions
                        while math.isclose(render_settings.apply_crop(part), 1.0, rel_tol=1e-2):
                            valid_locations.append(part.location.copy())
                            part.location[1] += conveyor_speed * np.sign(cam.location[1])

                # If there are 100 valid locations and pics_per_iter is 5, take 20 steps forward after each pic
                # Step size can be a float to ensure the entire range of valid locations is covered.
                if pics_per_iter == 1:
                    step = len(valid_locations)
                else:
                    step = (len(valid_locations) - 1) / (min(len(valid_locations), pics_per_iter) - 1)

                j = 0.0
                i = 0
                # Step through valid part locations with float step size. The actual index is rounded to the nearest integer
                while round(j) < len(valid_locations):
                    # set part location
                    part.location = valid_locations[round(j)]

                    # apply settings
                    render_settings.apply(part)

                    # render, output image file name is: partname_iternum_picnum.png
                    out_name = name + '_' + str(to_render[x]) + '_' + str(i)
                    bpy.data.scenes[0].render.filepath = OUT_PATH + out_name
                    bpy.ops.render.render(write_still=True)

                    # manage logs
                    if not isinstance(row[-1], dict):
                        row.append({})
                    vec = part.location - cam.location
                    vec = tuple(round(i, 1) for i in vec)
                    row[-1][out_name] = vec

                    j += step
                    i += 1

            # reset to defaults
            part.data.materials[0].node_tree.nodes["Group"].inputs[0].default_value = def_col
            for light in lights:
                light.data.color = light_def_vals[light.name]['color']
                light.data.energy = light_def_vals[light.name]['energy']
                light.location = light_def_vals[light.name]['location']

            amount -= 1

    # Write logs
    i = 0
    header.append(f'Pos Vectors rel. to Cam {str(tuple(round(i, 2) for i in cam.location))}')
    with open(OUT_CSV, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(header)
        for row in rendered_parts:
            writer.writerow(row)
            i += 1

    print(('--- Generated %s images of %s objects in ' + seconds_to_time(time.time() - t1) + ' ---')
          % (i * pics_per_iter * iters_per_part, i))

    if len(unrenderable_parts) > 0:
        print('The following parts could not be rendered:', *unrenderable_parts, sep='\n')

    if len(not_found_parts) > 0:
        print('The following parts could not be found:', *not_found_parts, sep='\n')


if __name__ == '__main__':
    exclude_list = ["75347.dat", "73485.dat", "2844.dat", "48288.dat", "4178523.dat", "2614.dat", "633.dat",
                    "55167.dat", "4107073.dat"]

    colors = [(254, 205, 4), (245, 124, 31), (221, 26, 34), (233, 94, 162), (255, 245, 121), (246, 172, 205),
              (251, 171, 24),
              (0, 108, 183), (0, 163, 218), (204, 224, 152), (0, 176, 78), (154, 201, 59), (106, 46, 20),
              (222, 139, 95),
              (244, 244, 244), (230, 237, 206), (149, 117, 180), (72, 158, 207), (0, 190, 212), (192, 228, 218),
              (221, 196, 142),
              (253, 195, 158), (188, 165, 207), (120, 191, 233), (112, 148, 122), (149, 126, 95), (160, 160, 158),
              (66, 67, 62), (75, 47, 147),
              (103, 130, 151), (0, 146, 71), (129, 131, 82), (174, 116, 70), (165, 83, 33), (101, 103, 102),
              (195, 151, 56), (0, 57, 94),
              (0, 75, 45), (58, 24, 14), (0, 0, 0), (135, 140, 143), (141, 5, 3), (4, 4, 4)]

    settings = RenderSettings(device='GPU',
                              crop_mode='part',
                              random_crop_pad_std=0.002,
                              samples=64,
                              max_bounces=4,
                              diffuse_bounces=3,
                              glossy_bounces=3,
                              transmission_bounces=3,
                              transparent_max_bounces=3,
                              use_persistent_data=True,
                              resolution_x=1920,
                              resolution_y=1080,
                              tile_size=2048)

    cam_settings = RaspiCamV1(pos_y=-10.5,
                              height=3.5,
                              height_range=0.2,
                              rot=55,
                              rot_range=0.3)

    generate_dataset(part_csv=ROOT + r'dataset\MostCommon.csv',
                     room_path=ROOT + r'dataset\blender\ConveyorRoom.fbx',
                     amount=1,
                     exclude_list=[],
                     colors=colors,
                     render_settings=settings,
                     iters_per_part=1,
                     pics_per_iter=5,
                     overwrite_existing=True,

                     conveyor_speed=0.2,
                     simulate_physics=True,
                     start_pos_z=3.5,
                     start_pos_x_std=0.5,
                     start_pos_y_std=0.8,

                     light_energy=1000,
                     cam_settings=cam_settings,
                     min_dist_to_cam=math.tan(math.radians(cam_settings.rot)) * cam_settings.height + 0.5)

