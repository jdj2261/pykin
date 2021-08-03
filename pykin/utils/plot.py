import numpy as np
import matplotlib.animation
import matplotlib.pyplot as plt
from pykin.kinematics import transformation as tf
# Colors of each directions axes. For ex X is green
directions_colors = ["green", "cyan", "orange"]


def plot_basis(ax, arm_length=1):
    """Plot a frame fitted to the robot size"""

    ax.set_xlim3d([-1.0, 1.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-1.0, 1.0])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-1.0, 1.0])
    ax.set_zlabel('Z')

    # Plot du repère
    # Sa taille est relative à la taille du bras
    ax.plot([0, arm_length * 1.5], [0, 0], [0, 0],
            c=directions_colors[0], label="X")
    ax.plot([0, 0], [0, arm_length * 1.5], [0, 0],
            c=directions_colors[1], label="Y")
    ax.plot([0, 0], [0, 0], [0, arm_length * 1.5],
            c=directions_colors[2], label="Z")


def plot_robot(robot, thetas, ax, name=None):
    links = []
    nodes = []
    transformation_matrix = []
    rotation_axes = []
    translation_axes = []

    transformations = robot.forward_kinematics(thetas)
    for (joint, (link, transformation)) in zip(robot.joints, transformations.items()):
        links.append(link)
        transformation_matrix.append(
            tf.get_homogeneous_matrix(transformation.pos, transformation.rot))
        
        # if joint.dtype == 'revloute':
        #     rotation_axes = joint.

    # for link, value in transformations.items():
    #     links.append(link)
    #     transformation_matrix.append(
    #         tf.get_homogeneous_matrix(value.pos, value.rot))
    
    cnt = 0
    for link, matrix in zip(links, transformation_matrix):
        # print(cnt, link, np.around(matrix, decimals=4))
        cnt += 1
        nodes.append(tf.get_pos_mat_from_homogeneous(matrix))

    if name == "baxter":
        torso_nodes = [nodes[0], nodes[3]]
        head_nodes = torso_nodes + nodes[7:12]
        pedestal_nodes = torso_nodes + [nodes[6]]
        right_nodes = torso_nodes + nodes[13:18] + nodes[20:31]
        left_nodes = torso_nodes + nodes[31:36] + nodes[38:47]

        head_lines = ax.plot([x[0] for x in head_nodes], [x[1] for x in head_nodes], [
            x[2] for x in head_nodes], linewidth=5, label="head")
        pedestal_lines = ax.plot([x[0] for x in pedestal_nodes], [x[1] for x in pedestal_nodes], [
            x[2] for x in pedestal_nodes], linewidth=5, label="pedestal")
        right_lines = ax.plot([x[0] for x in right_nodes], [x[1] for x in right_nodes], [
            x[2] for x in right_nodes], linewidth=5, label="right arm")
        left_lines = ax.plot([x[0] for x in left_nodes], [x[1] for x in left_nodes], [
            x[2] for x in left_nodes], linewidth=5, label="left arm")


        # for x in left_nodes:
        #     label = '(%0.2f, %0.2f, %0.2f)' % (x[0], x[1], x[2])
        #     ax.text(x[0], x[1], x[2], label, size="8")
        

        label = '(%0.2f, %0.2f, %0.2f)' % (
        left_nodes[-1][0], left_nodes[-1][1], left_nodes[-1][2])
        ax.text(left_nodes[-1][0], left_nodes[-1][1],
                left_nodes[-1][2], label, size="8")

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
            x[2] for x in nodes], linewidth=5, label="head")
        ax.scatter([x[0] for x in nodes], [x[1] for x in nodes],
        [x[2] for x in nodes], s=55, c=lines[0].get_color())

def plot_right_arm(robot, homogeneouses, ax, name=None):

    nodes = []
    for homogeneous in homogeneouses:
        nodes.append(tf.get_pos_mat_from_homogeneous(homogeneous))
    print(nodes)
    right_lines = ax.plot([x[0] for x in nodes], [x[1] for x in nodes], [
        x[2] for x in nodes], linewidth=5, label="right_arm")

    ax.scatter([x[0] for x in nodes], [x[1] for x in nodes], [x[2]
               for x in nodes], s=55, c=right_lines[0].get_color())


def init_3d_figure():
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_basis(ax)
    return fig, ax


def show_figure():
    plt.show()


