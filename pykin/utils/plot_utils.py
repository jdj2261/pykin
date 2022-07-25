import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pykin.utils import transform_utils as t_utils


def init_3d_figure(name=None, figsize=(12,8), dpi=100, visible_axis=False):
    """
    Initializes 3d figure
    """
    fig = plt.figure(name, figsize=figsize, dpi= dpi)
    ax = fig.add_subplot(111, projection='3d')

    if not visible_axis:
        ax.axis('off')
    fig.set_facecolor('beige')
    ax.set_facecolor('beige') 
    return fig, ax


def show_figure():
    """
    Show figure
    """
    plt.show()



def plot_basis(ax=None, robot=None):
    """
    Plot a frame fitted to the robot size
    """
    if robot is not None:
        offset = np.linalg.norm(robot.offset.pos)
    else:
        offset = 0.5
    
    if offset == 0:
        offset = 1

    ax.view_init(20,-10,)
    ax.set_xlim3d([-offset, offset])
    ax.set_ylim3d([-offset, offset])
    ax.set_zlim3d([-offset, offset])


def plot_robot(
    ax=None,
    robot=None,
    geom="collision",
    only_visible_geom=False, 
    visible_text=True,
    visible_scatter=True,
    alpha=1.0,
    color=None
):
    def _plot_baxter(ax, nodes, visible_text=True, visible_scatter=True):
        """
        Plot baxter robot
        """
        torso_nodes = [nodes[0]] + [nodes[3]]
        head_nodes = torso_nodes + nodes[7:12]
        pedestal_nodes = torso_nodes + [nodes[6]]
        right_nodes = torso_nodes + nodes[13:18] + nodes[20:29]
        left_nodes = torso_nodes + nodes[31:36] + nodes[38:47]

        head_lines = ax.plot([x[0] for x in head_nodes], [x[1] for x in head_nodes], [
            x[2] for x in head_nodes], linewidth=5, label="head")
        pedestal_lines = ax.plot([x[0] for x in pedestal_nodes], [x[1] for x in pedestal_nodes], [
            x[2] for x in pedestal_nodes], linewidth=5, label="pedestal")
        right_lines = ax.plot([x[0] for x in right_nodes], [x[1] for x in right_nodes], [
            x[2] for x in right_nodes], linewidth=5, label="right arm")
        left_lines = ax.plot([x[0] for x in left_nodes], [x[1] for x in left_nodes], [
            x[2] for x in left_nodes], linewidth=5, label="left arm")

        if visible_text:
            head_label = '(%0.4f, %0.4f, %0.4f)' % (
                head_nodes[-1][0], head_nodes[-1][1], head_nodes[-1][2])
            pedestal_label = '(%0.4f, %0.4f, %0.4f)' % (
                pedestal_nodes[-1][0], pedestal_nodes[-1][1], pedestal_nodes[-1][2])
            right_label = '(%0.4f, %0.4f, %0.4f)' % (
                right_nodes[8][0], right_nodes[8][1], right_nodes[8][2])
            left_label = '(%0.4f, %0.4f, %0.4f)' % (
                left_nodes[8][0], left_nodes[8][1], left_nodes[8][2])

            ax.text(head_nodes[-1][0], head_nodes[-1][1],
                    head_nodes[-1][2], head_label, size="8")
            ax.text(pedestal_nodes[-1][0], pedestal_nodes[-1][1],
                pedestal_nodes[-1][2], pedestal_label, size="8")
            ax.text(right_nodes[-1][0], right_nodes[-1][1],
                    right_nodes[-1][2], right_label, size="8")
            ax.text(left_nodes[-1][0], left_nodes[-1][1],
                    left_nodes[-1][2], left_label, size="8")

        if visible_scatter:
            ax.scatter([x[0] for x in head_nodes], [x[1] for x in head_nodes], 
                [x[2] for x in head_nodes], s=30, c=head_lines[0].get_color())
            ax.scatter([x[0] for x in pedestal_nodes], [x[1] for x in pedestal_nodes], 
                [x[2] for x in pedestal_nodes], s=30, c=pedestal_lines[0].get_color())
            ax.scatter([x[0] for x in right_nodes], [x[1] for x in right_nodes], 
                [x[2] for x in right_nodes], s=30, c=right_lines[0].get_color())
            ax.scatter([x[0] for x in left_nodes], [x[1] for x in left_nodes], 
                [x[2] for x in left_nodes], s=30, c=left_lines[0].get_color())
    """
    Plot robot
    """
    name = robot.robot_name
    
    plot_basis(ax, robot)

    if only_visible_geom:
        plot_geom(ax, robot, geom, alpha=alpha, color=color)
        
        if robot.has_gripper and robot.gripper.is_attached:
            plot_attached_object(ax, robot, alpha)
        return

    if robot.has_gripper and robot.gripper.is_attached:
        plot_attached_object(ax, robot, alpha)
                    
    links = []
    nodes = []
    transformation_matrix = []

    for i, (link, info) in enumerate(robot.info[geom].items()):
        if name != "baxter":
            if "pedestal" in link or "controller_box" in link or "tcp" in link:
                continue
        links.append(link)
        transformation_matrix.append(info[3])

    eef_idx = 0
    for i, (link, matrix) in enumerate(zip(links, transformation_matrix)):
        nodes.append(t_utils.get_pos_mat_from_homogeneous(matrix))
        if link == robot.eef_name:
            eef_idx=i

    if name == "baxter":
        _plot_baxter(ax, nodes, visible_text, visible_scatter)
    else:
        lines = ax.plot([x[0] for x in nodes], [x[1] for x in nodes], [
            x[2] for x in nodes], linewidth=2, label=name)

        if visible_text:
            label = '(%0.6f, %0.6f, %0.6f)' % (
                nodes[eef_idx][0], nodes[eef_idx][1], nodes[eef_idx][2])

            ax.text(nodes[eef_idx][0], nodes[eef_idx][1],
                    nodes[eef_idx][2], label, size="8")
        
        if visible_scatter:
            ax.scatter([x[0] for x in nodes], [x[1] for x in nodes],
                [x[2] for x in nodes], s=20, c=lines[0].get_color())

def plot_attached_object(ax, robot, alpha):
    plot_mesh(
        ax, 
        mesh=robot.gripper.info[robot.gripper.attached_obj_name][2], 
        h_mat=robot.gripper.info[robot.gripper.attached_obj_name][3], 
        alpha=alpha,
        color=robot.gripper.info[robot.gripper.attached_obj_name][4])

def plot_geom(ax, robot, geom="collision", alpha=0.4, color=None):
    """
    Plot robot's collision
    """

    plot_basis(ax, robot)
    for link, info in robot.info[geom].items():
        plot_geom_from_info(ax, robot, link, geom, info, alpha, color)

def plot_geom_from_info(ax, robot, link, geom, info, alpha, color):
    h_mat = info[3]
    if info[1] == 'mesh':
        meshes = np.array([info[2]]).reshape(-1)
        for idx, mesh in enumerate(meshes):
            mesh_color = get_mesh_color(robot, link, geom, idx, color)
            plot_mesh(ax, mesh=mesh, h_mat=h_mat, alpha=alpha, color=mesh_color)

    if info[1] == 'cylinder':
        for idx, param in enumerate(info[2]):
            length = float(param[0])
            radius = float(param[1])
            cylinder_color = get_color(robot.links[link].visual.gparam, idx)
            plot_cylinder(ax, length=length, radius=radius, h_mat=h_mat, alpha=alpha, color=cylinder_color)

    if info[1] == 'sphere':
        for idx, param in enumerate(info[2]):
            length = float(param)
            radius = float(param)
            pos = h_mat[:3,-1]
            sphere_color = get_color(robot.links[link].visual.gparam, idx)
            plot_sphere(ax, radius=radius, center_point=pos, n_steps=20, alpha=alpha, color=sphere_color)
    
    if info[1] == 'box':
        for idx, param in enumerate(info[2]):
            size = param
            box_color = get_color(robot.links[link].visual.gparam, idx)
            plot_box(ax, size, h_mat=h_mat, alpha=alpha, color=box_color)

def plot_objects(ax, objects, alpha=0.5):    
    """
    Plot objects
    """
    for _, info in objects.items():
        o_type = info.gtype
        o_param = info.gparam
        o_pose = info.h_mat
        if o_type == "mesh":
            for obj_name in ["table", "tray", "shelf_8", "shelf_9", "shelf_15", "clearbox"]:
                if obj_name in info.name:
                    _alpha = 0.2
                    break
                else:
                    _alpha = alpha
            plot_mesh(ax, mesh=o_param, h_mat=o_pose, alpha=_alpha, color=info.color)
        if o_type == "sphere":
            plot_sphere(ax, radius=o_param, center_point=o_pose, alpha=alpha, color=info.color)
        if o_type == "box":
            h_mat = t_utils.get_h_mat(o_pose)
            plot_box(ax, size=o_param, h_mat=h_mat, alpha=alpha, color=info.color)
        if o_type == "cylinder":
            h_mat = t_utils.get_h_mat(o_pose)
            plot_cylinder(ax, radius=o_param[0], length=o_param[1], h_mat=h_mat, n_steps=100, alpha=alpha, color=info.color)


def plot_object(ax, obj, pose=None, alpha=0.5):    
    """
    Plot objects
    """
    o_type = obj.gtype
    o_param = obj.gparam
    if pose is None:
        o_pose = obj.h_mat
    else:
        o_pose = pose
    if o_type == "mesh":
        plot_mesh(ax, mesh=o_param, h_mat=o_pose, alpha=alpha, color=obj.color)
    if o_type == "sphere":
        plot_sphere(ax, radius=o_param, center_point=o_pose, alpha=alpha, color=obj.color)
    if o_type == "box":
        h_mat = t_utils.get_h_mat(o_pose)
        plot_box(ax, size=o_param, h_mat=h_mat, alpha=alpha, color=obj.color)
    if o_type == "cylinder":
        h_mat = t_utils.get_h_mat(o_pose)
        plot_cylinder(ax, radius=o_param[0], length=o_param[1], h_mat=h_mat, n_steps=100, alpha=alpha, color=obj.color)


def render_axis(
        ax,
        pose,
        axis=[1, 1, 1],
        scale=0.05
    ):
        if axis[0]:
            plot_normal_vector(ax, pose[:3, 3], pose[:3, 0], scale=scale, edgecolor="red")
        if axis[1]:
            plot_normal_vector(ax, pose[:3, 3], pose[:3, 1], scale=scale, edgecolor="green")
        if axis[2]:
            plot_normal_vector(ax, pose[:3, 3], pose[:3, 2], scale=scale, edgecolor="blue")


def get_mesh_color(robot, link, geom, idx=0, color=None):
    mesh_color = color
    if color is None:
        mesh_color = np.array([0.2, 0.2, 0.2, 1])
        robot_link = robot.links.get(link)
        if geom == "collision":
            if robot_link:
                if robot_link.collision.gparam.get('color'):
                    mesh_color = robot_link.collision.gparam.get('color')[idx]
                    mesh_color = np.array([color for color in mesh_color.values()]).flatten()
            else: 
                if robot.has_gripper:
                    if robot.gripper.info.get(robot.gripper.attached_obj_name) is not None:
                        mesh_color = robot.gripper.info.get(robot.gripper.attached_obj_name)[4]
        else:
            if robot_link:
                if robot_link.visual.gparam.get('color'):
                    mesh_color = robot_link.visual.gparam.get('color')[idx]
                    mesh_color = np.array([color for color in mesh_color.values()]).flatten()
                else: 
                    if robot.has_gripper:
                        if robot.gripper.info.get(robot.gripper.attached_obj_name) is not None:
                            mesh_color = robot.gripper.info.get(robot.gripper.attached_obj_name)[4]
    return mesh_color

def get_color(params, idx=0):
    def convert_color_type(color):
        if isinstance(color, str):
            color = color
        
        if isinstance(color, list):
            if len(color) == 0:
                color = 'k'
            else:
                color = color[0]
        
        if isinstance(color, np.ndarray):
            if len(color) == 0:
                color = np.array([0.2, 0.2, 0.2, 1.])
            else:
                color = color

        if isinstance(color, dict):
            if len(color) == 0:
                color = np.array([0.2, 0.2, 0.2, 1.])
            else:
                color = list(color.values())[0]
        return color
        
    color = []
    if params is not None and params.get('color'):
        visual_color = params.get('color')[idx]
        if visual_color is not None:
            color = list(visual_color.values())
    color = convert_color_type(color)
    return color
    

def plot_cylinder(
    ax=None, 
    length=1.0, 
    radius=1.0,
    h_mat=np.eye(4), 
    n_steps=100,
    alpha=1.0, 
    color="k"
):
    """
    Plot cylinder
    """
    axis_start = h_mat.dot(np.array([0, 0, -length/2, 1]))[:3]
    axis_end =  h_mat.dot(np.array([0, 0, length/2, 1]))[:3]

    axis = axis_end - axis_start
    axis /= length
    not_axis = np.array([1, 0, 0])
    if (axis == not_axis).all():
        not_axis = np.array([0, 1, 0])

    n1 = np.cross(axis, not_axis)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(axis, n1)

    t = np.array([0, length])

    theta = np.linspace(0, 2 * np.pi, n_steps)
    t, theta = np.meshgrid(t, theta)

    X, Y, Z = [axis_start[i] + axis[i] * t
            + radius * np.sin(theta) * n1[i]
            + radius * np.cos(theta) * n2[i] for i in [0, 1, 2]]

    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0)


def plot_sphere(
    ax=None, 
    radius=1.0, 
    center_point=np.zeros(3), 
    n_steps=20, 
    alpha=1.0, 
    color="k"
):
    """
    Plot sphere
    """
    phi, theta = np.mgrid[0.0:np.pi:n_steps * 1j, 0.0:2.0 * np.pi:n_steps * 1j]
    x = center_point[0] + radius * np.sin(phi) * np.cos(theta)
    y = center_point[1] + radius * np.sin(phi) * np.sin(theta)
    z = center_point[2] + radius * np.cos(phi)

    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)


def plot_box(ax=None, size=np.ones(3), alpha=1.0, h_mat=np.eye(4), color="k"):
    """
    Plot box
    """
    if not isinstance(size, np.ndarray):
        size = np.array(size)

    corners = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])
    corners = (corners - 0.5) * size
    PA = np.hstack(
        (corners, np.ones((len(corners), 1))))

    corners = np.dot(PA, h_mat.T)[:, :3]
    p3c = Poly3DCollection(np.array([
        [corners[0], corners[1], corners[2]],
        [corners[1], corners[2], corners[3]],

        [corners[4], corners[5], corners[6]],
        [corners[5], corners[6], corners[7]],

        [corners[0], corners[1], corners[4]],
        [corners[1], corners[4], corners[5]],

        [corners[2], corners[6], corners[7]],
        [corners[2], corners[3], corners[7]],

        [corners[0], corners[4], corners[6]],
        [corners[0], corners[2], corners[6]],

        [corners[1], corners[5], corners[7]],
        [corners[1], corners[3], corners[7]],
    ]))

    p3c.set_alpha(alpha)
    p3c.set_facecolor(color)
    ax.add_collection3d(p3c)


def plot_mesh(
    ax=None, 
    mesh=None, 
    h_mat=np.eye(4),
    s=np.array([1.0, 1.0, 1.0]),
    alpha=1.0, 
    color="k"):
    vertices = mesh.vertices * s
    vertices = np.hstack((vertices, np.ones((len(vertices), 1))))
    vertices = np.dot(vertices, h_mat.T)[:, :3]
    vectors = np.array([vertices[[i, j, k]] for i, j, k in mesh.faces])

    surface = Poly3DCollection(vectors)
    surface.set_facecolor(color)
    surface.set_alpha(alpha)
    ax.add_collection3d(surface)


def plot_normal_vector(ax, vertices, normals, scale=1, linewidths=(1,), edgecolor="red"):
    
    if vertices.ndim != 2:
        vertices = vertices.reshape(1, -1)
    if normals.ndim != 2:
        normals = normals.reshape(1, -1)

    ax.quiver(
        [vertex[0] for vertex in vertices], 
        [vertex[1] for vertex in vertices], 
        [vertex[2] for vertex in vertices], 
        [normal[0]*scale for normal in normals], 
        [normal[1]*scale for normal in normals], 
        [normal[2]*scale for normal in normals], linewidths=linewidths, edgecolor=edgecolor)    


def plot_axis(
    ax,
    pose,
    axis=[1, 1, 1],
    scale=0.1
):
    if axis[0]:
        plot_normal_vector(ax, pose[:3, 3], pose[:3, 0], scale=scale, edgecolor="red")
    if axis[1]:
        plot_normal_vector(ax, pose[:3, 3], pose[:3, 1], scale=scale, edgecolor="green")
    if axis[2]:
        plot_normal_vector(ax, pose[:3, 3], pose[:3, 2], scale=scale, edgecolor="blue")


def plot_vertices(ax, vertices, s=5, c='k'):
    if vertices.ndim != 2:
        vertices = vertices.reshape(1, -1)
    ax.scatter([x[0] for x in vertices], [x[1] for x in vertices], 
        [x[2] for x in vertices], s=s, c=c)


def plot_line(ax, vertices, linewidth=1):
    if vertices.ndim != 2:
        vertices = vertices.reshape(1, -1)
    ax.plot(
        [x[0] for x in vertices], 
        [x[1] for x in vertices],
        [x[2] for x in vertices], linewidth=linewidth)


def plot_path_planner(ax, path):
    """
    Plot rrt* path planner
    """
    if path is None:
        print("cannot create path")
        return

    ax.scatter([x for (x, y, z) in path], [y for (x, y, z) in path], [z for (x, y, z) in path], s=10, c='r')
    ax.plot([x for (x, y, z) in path], [y for (x, y, z) in path], [z for (x, y, z) in path], '-b', linewidth=0.5,)
    ax.text(path[0][0], path[0][1], path[0][2], 'Start', verticalalignment='bottom', horizontalalignment='center', size="20")
    ax.text(path[-1][0], path[-1][1], path[-1][2],'Goal', verticalalignment='bottom', horizontalalignment='center', size="20")


def plot_trajectories(ax, path, size=3, color='r'):
    """
    Plot plot_trajectories
    """
    ax.scatter([x for (x, y, z) in path], [y for (x, y, z) in path], [z for (x, y, z) in path], s=size, c=color)
