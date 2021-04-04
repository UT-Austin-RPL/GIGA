import pickle
import pybullet
import numpy as np
from vgn.utils.transform import Rotation, Transform

def get_mesh_pose_dict_from_world(world,
                                  physicsClientId=0,
                                  exclude_plane=True):
    mesh_pose_dict = {}
    for uid in world.bodies.keys():
        _, name = world.p.getBodyInfo(uid)
        name = name.decode('utf8')
        if name == 'plane' and exclude_plane:
            continue
        body = world.bodies[uid]
        for visual in world.p.getVisualShapeData(uid, physicsClientId):
            object_name, mesh_path, scale, pose = get_mesh_pose(
                visual, physicsClientId)
            mesh_path = mesh_path.decode('utf8')
            mesh_pose_dict[object_name] = (mesh_path, scale, pose)
    return mesh_pose_dict

def get_mesh_pose(visual, physicsClientId):
    objectUniqueId = visual[0]
    linkIndex = visual[1]
    visualGeometryType = visual[2]
    scale = visual[3]
    meshAssetFileName = visual[4]
    localVisualFramePosition = visual[5]
    localVisualFrameOrientation = visual[6]
    rgbaColor = visual[7]

    visual_offset = Transform(Rotation.from_quat(localVisualFrameOrientation),
                              np.array(localVisualFramePosition))
    if linkIndex != -1:
        linkState = get_link_pose((objectUniqueId, linkIndex), physicsClientId)
        linkOffsetState = get_link_local_offset((objectUniqueId, linkIndex), physicsClientId)
        linkOffsetState.translation = np.array([0, 0, 0])
        linkOffsetState = linkOffsetState * visual_offset
    else:
        linkState = get_body_pose(objectUniqueId, physicsClientId)
        linkOffsetState = visual_offset
#     transform = linkState
    transform = linkState * linkOffsetState
    
    # Name to use for visii components
    object_name = str(objectUniqueId) + "_" + str(linkIndex)
    return object_name, meshAssetFileName, scale, transform

def get_link_local_offset(link_uid, physicsClientId=0):
    """Get the local offset of the link.
    Args:
        link_uid: A tuple of the body Unique ID and the link index.
    Returns:
        An instance of Pose.
    """
    body_uid, link_ind = link_uid
    _, _, position, quaternion, _, _ = pybullet.getLinkState(
        bodyUniqueId=body_uid, linkIndex=link_ind, physicsClientId=physicsClientId)
    ori = Rotation.from_quat(quaternion)
    return Transform(ori, np.array(position))

def get_link_pose(link_uid, physicsClientId):
    """Get the pose of the link.
    Args:
        link_uid: A tuple of the body Unique ID and the link index.
    Returns:
        An instance of Pose.
    """
    body_uid, link_ind = link_uid
    _, _, _, _, position, quaternion = pybullet.getLinkState(
        bodyUniqueId=body_uid, linkIndex=link_ind,
        physicsClientId=physicsClientId)
    ori = Rotation.from_quat(quaternion)
    return Transform(ori, np.array(position))

def get_link_center_pose(link_uid, physicsClientId):
    """Get the pose of the link center of mass.
    Args:
        link_uid: A tuple of the body Unique ID and the link index.
    Returns:
        An instance of Pose.
    """
    body_uid, link_ind = link_uid
    position, quaternion, _, _, _, _ = pybullet.getLinkState(
        bodyUniqueId=body_uid, linkIndex=link_ind,
        physicsClientId=physicsClientId)
    ori = Rotation.from_quat(quaternion)
    return Transform(ori, np.array(position))

def get_body_pose(body_uid, physicsClientId):
    """Get the pose of the body.
    The pose of the body is defined as the pose of the base of the body.
    Args:
        body_uid: The body Unique ID.
    Returns:
        An instance of Pose.
    """
    position, quaternion = pybullet.getBasePositionAndOrientation(
        bodyUniqueId=body_uid, physicsClientId=physicsClientId)

    tmp = pybullet.getDynamicsInfo(body_uid, -1, physicsClientId)
    local_inertial_pos, local_inertial_ori = tmp[3], tmp[4]
    local_transform = Transform(Rotation.from_quat(local_inertial_ori),
                                np.array(local_inertial_pos))

    base_frame_transform = Transform(
        Rotation.from_quat(quaternion),
        np.array(position)) * local_transform.inverse()

    return base_frame_transform