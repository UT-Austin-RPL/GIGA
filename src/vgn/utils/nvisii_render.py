import os
import glob
import random
import nvisii
from scipy.spatial.transform import Rotation as R


class NViSIIRenderer:
    def __init__(self, opt):
        self.opt = opt
        self.spp = opt['spp']
        self.width = opt['width']
        self.height = opt['height']

    def reset(self):
        nvisii.clear_all()
        # Camera
        self.camera = nvisii.entity.create(name="camera")
        self.camera.set_transform(
            nvisii.transform.create(name="camera_transform"))
        self.camera.set_camera(
            nvisii.camera.create_from_fov(
                name="camera_camera",
                field_of_view=0.785398,  # note, this is in radians
                aspect=self.width / float(self.height)))
        nvisii.set_camera_entity(self.camera)
        self.set_camera(self.opt['camera']['position'],
                        self.opt['camera']['look_at'])

        # Light
        self.light = nvisii.entity.create(
            name="light_0",
            mesh=nvisii.mesh.create_plane("light_0", flip_z=True),
            transform=nvisii.transform.create("light_1"),
            light=nvisii.light.create("light_1"))
        self.set_light(self.opt['light']['intensity'],
                       self.opt['light']['scale'],
                       self.opt['light']['position'],
                       self.opt['light']['look_at'])

        # Floor
        self.floor = nvisii.entity.create(
            name="floor",
            mesh=nvisii.mesh.create_plane("mesh_floor"),
            transform=nvisii.transform.create("transform_floor"),
            material=nvisii.material.create("material_floor"))
        self.set_floor(self.opt['floor']['texture'],
                       self.opt['floor']['scale'],
                       self.opt['floor']['position'])

        self.objects = {}


    def place_objects_from_list(self, mesh_pose_list, object_root=None):
        # new_objects = []
        # removed_objects = []
        if object_root is not None:
            object_files = glob.glob(object_root + '/*')

        for idx, (path, scale, pose_matrix) in enumerate(mesh_pose_list):
            if object_root is not None:
                path = object_files[random.randint(0, len(object_files) - 1)]
                mesh = nvisii.mesh.create_from_file('mesh_' + str(idx), path + '/meshes/model.obj')
                obj_entity = nvisii.entity.create(
                    name="entity" + str(idx),
                    mesh = mesh,
                    transform = nvisii.transform.create("transform" + str(idx)),
                    material = nvisii.material.create("material" + str(idx))
                )
                obj_entity.get_transform().set_position(pose_matrix[:3,3])
                obj_entity.get_transform().set_rotation(R.from_matrix(pose_matrix[0:3, 0:3]).as_quat())
                obj_entity.get_transform().set_scale((scale, scale, scale))

                obj_texture = nvisii.texture.create_from_file(
                        name=str(idx), path=path + '/materials/textures/texture.png', linear=True)

                obj_entity.get_material().set_base_color_texture(obj_texture)
            else:
                obj = nvisii.import_scene(path)
                obj.transforms[0].set_position(pose_matrix[:3,3])
                obj.transforms[0].set_rotation(R.from_matrix(pose_matrix[0:3, 0:3]).as_quat())
                obj.transforms[0].set_scale((scale, scale, scale))


    def update_objects(self, mesh_pose_dict):
        new_objects = []
        removed_objects = []
        for k, (path, scale, transform) in mesh_pose_dict.items():
            if k not in self.objects.keys():
                obj = nvisii.import_scene(path)
                obj.transforms[0].set_position(transform.translation)
                obj.transforms[0].set_rotation(transform.rotation.as_quat())
                obj.transforms[0].set_scale(scale)
                self.objects[k] = obj
                new_objects.append(k)
            else:
                obj = self.objects[k]
                obj.transforms[0].set_position(transform.translation)
                obj.transforms[0].set_rotation(transform.rotation.as_quat())
                obj.transforms[0].set_scale(scale)
        for k in self.objects.keys():
            if k not in mesh_pose_dict.keys():
                for obj in self.objects[k].entities:
                    obj.remove(obj.get_name())
                removed_objects.append(k)
        for k in removed_objects:
            self.objects.pop(k)

        return new_objects, removed_objects

    def render(self, path=None):
        if path is None:
            img = nvisii.render(width=self.width,
                        height=self.height,
                        samples_per_pixel=self.spp)
            return img
        else:
            nvisii.render_to_file(width=self.width,
                                height=self.height,
                                samples_per_pixel=self.spp,
                                file_path=path)

    def set_camera(self, position, look_at, up=(0, 0, 1)):
        self.camera.get_transform().set_position(position)
        self.camera.get_transform().look_at(at=look_at, up=(0, 0, 1))

    def set_light(self, intensity, scale, position, look_at, up=(0, 0, 1)):
        self.light.get_light().set_intensity(intensity)
        self.light.get_transform().set_scale(scale)
        self.light.get_transform().set_position(position)
        self.light.get_transform().look_at(at=look_at, up=(0, 0, 1))

    def set_floor(self, texture_path, scale, position):
        if hasattr(self, 'floor_texture'):
            self.floor_texture.remove(self.floor_texture.get_name())
        self.floor_texture = nvisii.texture.create_from_file(
            name='floor_texture', path=texture_path)
        self.floor.get_material().set_base_color_texture(self.floor_texture)
        self.floor.get_material().set_roughness(0.4)
        self.floor.get_material().set_specular(0)

        self.floor.get_transform().set_scale(scale)
        self.floor.get_transform().set_position(position)

    @staticmethod
    def init():
        nvisii.initialize(headless=True, lazy_updates=True, verbose=False)
        nvisii.enable_denoiser()

    @staticmethod
    def deinit():
        nvisii.deinitialize()