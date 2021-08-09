import os
import numpy as np
import matplotlib.animation
import matplotlib.pyplot as plt
from pykin.kinematics import transformation as tf
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


def plot_robot(robot, fk, ax, name=None):
    # TODO
    # nodes : list to dict (name, value)
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

    
def init_3d_figure(name=None):
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    fig = plt.figure(name)
    ax = fig.add_subplot(111, projection='3d')

    return fig, ax


def show_figure():
    plt.show()


