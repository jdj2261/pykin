import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pykin.utils import transform_utils as tf

# Colors of each directions axes. For ex X is green
directions_colors = ["green", "cyan", "orange"]

def init_3d_figure(name=None, figsize=(15,7.5), dpi= 80):
    """
    Initializes 3d figure
    """
    fig = plt.figure(name, figsize=figsize, dpi= dpi)
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax


def show_figure():
    """
    Show figure
    """
    plt.show()


def _check_color_type(color):
    """
    Check color's data type
    """
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


def plot_basis(ax=None, robot=None):
    """
    Plot a frame fitted to the robot size
    """
    if robot is not None:
        offset = np.linalg.norm(robot.offset.pos)
    else:
        offset = 1
        
    if offset == 0:
        offset = 1

    ax.set_xlim3d([-1.0 * offset, 1.0 * offset])
    ax.set_xlabel('X')

    ax.set_ylim3d([-1.0 * offset, 1.0 * offset])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-1.0 * offset, 1.0 * offset])
    ax.set_zlabel('Z')


    ax.plot([0, offset * 1.5], [0, 0], [0, 0],
            c=directions_colors[0], label="X")
    ax.plot([0, 0], [0, offset * 1.5], [0, 0],
            c=directions_colors[1], label="Y")
    ax.plot([0, 0], [0, 0], [0, offset * 1.5],
            c=directions_colors[2], label="Z")


def plot_robot(
    robot, 
    ax=None, 
    fk=None, 
    visible_collision=False, 
    visible_text=True,
    visible_scatter=True,
    visible_basis=True):

    """
    Plot robot
    """
    if fk is None:
        fk = robot.init_fk

    name = robot.robot_name

    if visible_basis:
        plot_basis(ax, robot)
    
    links = []
    nodes = []
    transformation_matrix = []

    for i, (link, transformation) in enumerate(fk.items()):
        links.append(link)
        transformation_matrix.append(transformation.h_mat)

    eef_idx = 0

    for i, (link, matrix) in enumerate(zip(links, transformation_matrix)):
        nodes.append(tf.get_pos_mat_from_homogeneous(matrix))
        if link == robot.eef_name:
            eef_idx=i

    if name == "baxter":
        plot_baxter(ax, nodes, visible_text, visible_scatter)
    else:
        lines = ax.plot([x[0] for x in nodes], [x[1] for x in nodes], [
            x[2] for x in nodes], linewidth=2, label=name)

        if visible_text:
            label = '(%0.4f, %0.4f, %0.4f)' % (
                nodes[eef_idx][0], nodes[eef_idx][1], nodes[eef_idx][2])

            ax.text(nodes[eef_idx][0], nodes[eef_idx][1],
                    nodes[eef_idx][2], label, size="8")
        
        if visible_scatter:
            ax.scatter([x[0] for x in nodes], [x[1] for x in nodes],
                [x[2] for x in nodes], s=20, c=lines[0].get_color())
    

    if visible_collision:
        plot_collision(ax, robot, fk)

    ax.legend()


def plot_baxter(ax, nodes, visible_text=True, visible_scatter=True):
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


def plot_trajectories(ax, path, size=10, color='r'):
    """
    Plot plot_trajectories
    """
    ax.scatter([x for (x, y, z) in path], [y for (x, y, z) in path], [z for (x, y, z) in path], s=size, c=color)


def plot_animation(
    robot, 
    trajectory,
    fig=None,
    ax=None,
    eef_poses=None,
    objects=None,
    visible_objects=False,
    visible_collision=False,
    visible_text=True,
    visible_scatter=True,
    interval=100, 
    repeat=False
    ):

    """
    Plot animation
    """

    def update(i):
        # print(f"{i/len(trajectory) * 100:.1f} %")
    
        if i == len(trajectory)-1:
            # print(f"{i/(len(trajectory)-1) * 100:.1f} %")
            print("Animation Finished..")
        ax.clear()

        if visible_objects and objects:
            plot_objects(ax, objects)
        
        if eef_poses is not None:
            plot_trajectories(ax, eef_poses)
          
        plot_robot(
            robot, 
            fk=trajectory[i], 
            ax=ax, 
            visible_collision=visible_collision,
            visible_text=visible_text,
            visible_scatter=visible_scatter)

    ani = animation.FuncAnimation(fig, update, np.arange(len(trajectory)), interval=interval, repeat=repeat)
    plt.show()


def plot_objects(ax, objects, alpha=0.5, color='k'):    
    """
    Plot objects
    """
    for key, value in objects:
        o_type = value[0]
        o_param = value[1]
        o_pose = value[2]

        if o_type == "mesh":
            plot_mesh(ax, mesh=o_param, h_mat=o_pose, alpha=alpha, color=color)
        if o_type == "sphere":
            plot_sphere(ax, radius=o_param, center_point=o_pose, alpha=alpha, color=color)
        if o_type == "box":
            h_mat = tf.get_h_mat(o_pose)
            plot_box(ax, size=o_param, h_mat=h_mat, alpha=alpha, color=color)
        if o_type == "cylinder":
            h_mat = tf.get_h_mat(o_pose)
            plot_cylinder(ax, radius=o_param[0], length=o_param[1], h_mat=h_mat, n_steps=100, alpha=alpha, color=color)


def plot_collision(ax, robot, fk, alpha=0.8):
    """
    Plot robot's collision
    """
    def _get_color(params):
        color = []
        if params is not None:
            visual_color = params.get('color')
            if visual_color is not None:
                color = list(visual_color.keys())
        return color

    for link, transformation in fk.items():
        h_mat = np.dot(transformation.h_mat, robot.links[link].collision.offset.h_mat)
        color = _get_color(robot.links[link].visual.gparam)

        if robot.links[link].collision.gtype == 'cylinder':
            length = float(robot.links[link].collision.gparam.get('length'))
            radius = float(robot.links[link].collision.gparam.get('radius'))
            plot_cylinder(ax, length=length, radius=radius, h_mat=h_mat, alpha=alpha, color=color)

        if robot.links[link].collision.gtype == 'sphere':
            radius = float(robot.links[link].collision.gparam.get('radius'))
            pos = h_mat[:3,-1]
            plot_sphere(ax, radius=radius, center_point=pos, n_steps=20, alpha=alpha, color=color)
    
        if robot.links[link].collision.gtype == 'box':
            size = robot.links[link].collision.gparam.get('size')
            plot_box(ax, size, h_mat=h_mat, alpha=alpha, color=color)


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
    color = _check_color_type(color)
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
    color = _check_color_type(color)
    phi, theta = np.mgrid[0.0:np.pi:n_steps * 1j, 0.0:2.0 * np.pi:n_steps * 1j]
    x = center_point[0] + radius * np.sin(phi) * np.cos(theta)
    y = center_point[1] + radius * np.sin(phi) * np.sin(theta)
    z = center_point[2] + radius * np.cos(phi)

    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)


def plot_box(ax=None, size=np.ones(3), alpha=1.0, h_mat=np.eye(4), color="k"):
    """
    Plot box
    """
    color = _check_color_type(color)
    
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
