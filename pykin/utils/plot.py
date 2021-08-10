import os
import warnings
import numpy as np
import matplotlib.animation
import matplotlib.pyplot as plt
from pykin.kinematics import transformation as tf
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Colors of each directions axes. For ex X is green
directions_colors = ["green", "cyan", "orange"]


def plot_basis(robot, ax, arm_length=1):
    """Plot a frame fitted to the robot size"""

    offset = np.linalg.norm(robot.offset.pos)

    if offset == 0:
        offset = 1

    ax.set_xlim3d([-1.0 * offset, 1.0 * offset])
    ax.set_xlabel('X')

    ax.set_ylim3d([-1.0 * offset, 1.0 * offset])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-1.0 * offset, 1.0 * offset])
    ax.set_zlabel('Z')

    # Plot du repère
    # Sa taille est relative à la taille du bras
    ax.plot([0, offset * 1.5], [0, 0], [0, 0],
            c=directions_colors[0], label="X")
    ax.plot([0, 0], [0, offset * 1.5], [0, 0],
            c=directions_colors[1], label="Y")
    ax.plot([0, 0], [0, 0], [0, offset * 1.5],
            c=directions_colors[2], label="Z")


def plot_robot(robot, fk, ax, name=None, visible_collision=True, mesh_path='../asset/urdf/baxter/'):

    if name is not None:
        name = os.path.splitext(os.path.basename(name))[0].strip()

    plot_basis(robot, ax)
    links = []
    nodes = []
    transformation_matrix = []

    for (link, transformation) in fk.items():
        links.append(link)
        transformation_matrix.append(
            tf.get_homogeneous_matrix(transformation.pos, transformation.rot))

    for link, matrix in zip(links, transformation_matrix):
        nodes.append(tf.get_pos_mat_from_homogeneous(matrix))

    if name == "baxter":
        torso_nodes = [nodes[0], nodes[3]]
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

        ax.scatter([x[0] for x in head_nodes], [x[1] for x in head_nodes], 
            [x[2] for x in head_nodes], s=55, c=head_lines[0].get_color())
        ax.scatter([x[0] for x in pedestal_nodes], [x[1] for x in pedestal_nodes], 
            [x[2] for x in pedestal_nodes], s=55, c=pedestal_lines[0].get_color())
        ax.scatter([x[0] for x in right_nodes], [x[1] for x in right_nodes], 
            [x[2] for x in right_nodes], s=55, c=right_lines[0].get_color())
        ax.scatter([x[0] for x in left_nodes], [x[1] for x in left_nodes], 
            [x[2] for x in left_nodes], s=55, c=left_lines[0].get_color())
    else:
        lines = ax.plot([x[0] for x in nodes], [x[1] for x in nodes], [
            x[2] for x in nodes], linewidth=5, label=name)

        label = '(%0.4f, %0.4f, %0.4f)' % (
            nodes[-1][0], nodes[-1][1], nodes[-1][2])

        ax.text(nodes[-1][0], nodes[-1][1],
                nodes[-1][2], label, size="8")
        
        ax.scatter([x[0] for x in nodes], [x[1] for x in nodes],
            [x[2] for x in nodes], s=55, c=lines[0].get_color())

    if visible_collision:
        for i, (link, transformation) in enumerate(fk.items()):
            if robot.links[i].dtype == 'cylinder':
                A2B = tf.get_homogeneous_matrix(
                    fk[robot.links[i].name].pos, fk[robot.links[i].name].rot)
                length = float(robot.links[i].length)
                radius = float(robot.links[i].radius)
                plot_cylinder(
                    robot.links[i], fk, ax, length=length, radius=radius, A2B=A2B, alpha=0.8, color=robot.links[i].color)
            if robot.links[i].dtype == 'sphere':
                radius = float(robot.links[i].radius)
                pos = fk[robot.links[i].name].pos
                plot_sphere(
                    robot.links[i], fk, ax, radius=radius, alpha=0.1, color=robot.links[i].color, p=pos, n_steps=20, ax_s=0.5)
            if robot.links[i].dtype == 'box':
                size = robot.links[i].size
                A2B = tf.get_homogeneous_matrix(
                    fk[robot.links[i].name].pos, fk[robot.links[i].name].rot)
                plot_box(robot.links[i], fk, ax, size, A2B=A2B,
                         alpha=0.6, color=robot.links[i].color)
        filename = mesh_path + 'meshes/base/pedestal_link_collision.stl'
        plot_mesh(ax, filename=filename, alpha=0.3)
        filename = mesh_path + 'meshes/torso/base_link.stl'
        plot_mesh(ax, filename=filename, alpha=0.3)
            # if robot.links[i].mesh is not None:
            #     filename = mesh_path + robot.links[i].mesh
            #     A2B = tf.get_homogeneous_matrix(
            #         fk[robot.links[i].name].pos, fk[robot.links[i].name].rot)
            #     plot_mesh(ax, filename=filename, A2B=A2B, alpha=0.3)
                

def plot_cylinder(link, fk, ax=None, length=1.0, radius=1.0,
                  A2B=np.eye(4), n_steps=100,
                  alpha=1.0, color="k"):

    A2B[:3,-1] = np.add(A2B[:3,-1],link.offset.pos)
    axis_start = A2B.dot(np.array([0, 0, -length, 1]))[:3]
    axis_end =  A2B.dot(np.array([0, 0, length, 1]))[:3]

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


def plot_sphere(link, fk, ax=None, radius=1.0, p=np.zeros(3), ax_s=1,
                n_steps=20, alpha=1.0, color="k"):

    phi, theta = np.mgrid[0.0:np.pi:n_steps * 1j, 0.0:2.0 * np.pi:n_steps * 1j]
    x = p[0] + radius * np.sin(phi) * np.cos(theta)
    y = p[1] + radius * np.sin(phi) * np.sin(theta)
    z = p[2] + radius * np.cos(phi)

    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)


def plot_box(link, fk, ax=None, size=np.ones(3), A2B=np.eye(4), ax_s=1,
             color="k", alpha=1.0):

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

    corners = np.dot(PA, A2B.T)[:, :3]
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


def plot_mesh(ax=None, filename=None, A2B=np.eye(4),
              s=np.array([1.0, 1.0, 1.0]), ax_s=1, wireframe=False,
              convex_hull=False, alpha=1.0, color="k"):

    try:
        import trimesh
    except ImportError:
        warnings.warn(
            "Cannot display mesh. Library 'trimesh' not installed.")
        return ax

    mesh = trimesh.load(filename)
    vertices = mesh.vertices * s
    vertices = np.hstack((vertices, np.ones((len(vertices), 1))))
    vertices = np.dot(vertices, A2B.T)[:, :3]
    vectors = np.array([vertices[[i, j, k]] for i, j, k in mesh.faces])

    surface = Poly3DCollection(vectors)
    surface.set_facecolor(color)
    surface.set_alpha(alpha)
    ax.add_collection3d(surface)
    

def init_3d_figure(name=None):
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    fig = plt.figure(name)
    ax = fig.add_subplot(111, projection='3d')

    return fig, ax


def show_figure():
    plt.show()


