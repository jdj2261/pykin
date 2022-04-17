import numpy as np
import pykin.utils.plot_utils as plt

class SceneRender:

    @staticmethod
    def render_all_scene(ax, objs, robot, alpha, robot_color):
        SceneRender.render_object(ax, objs, alpha)
        SceneRender.render_robot(ax, robot, alpha, robot_color)

    @staticmethod
    def render_object_and_gripper(ax, objs, robot, alpha, gripper_color, visible_tcp):
        SceneRender.render_object(ax, objs, alpha)
        SceneRender.render_gripper(ax, robot, alpha, gripper_color, visible_tcp)

    @staticmethod
    def render_object(ax, objs, alpha):
        plt.plot_basis(ax)
        for info in objs.values():
            plt.plot_mesh(
                ax=ax, 
                mesh=info.gparam, 
                h_mat=info.h_mat, 
                color=info.color,
                alpha=alpha,
            )

    @staticmethod
    def render_robot(ax, robot, alpha, color):
        plt.plot_basis(ax)
        for link, info in robot.info.items():
            if info[1] == 'mesh':
                mesh_color = color
                if color is None:
                    mesh_color = robot.links[link].collision.gparam.get('color')
                    mesh_color = np.array([color for color in mesh_color.values()]).flatten()
                if "finger" in link:
                    alpha = 1
                plt.plot_mesh(ax, mesh=info[2], h_mat=info[3], alpha=alpha, color=mesh_color)

    @staticmethod
    def render_gripper(ax, robot, alpha=0.3, color='b', visible_tcp=True):
        plt.plot_basis(ax)
        for link, info in robot.gripper.info.items():
            if info[1] == 'mesh':
                plt.plot_mesh(ax, mesh=info[2], h_mat=info[3], alpha=alpha, color=color)

        if visible_tcp:
            ax.scatter(
                robot.gripper.info["tcp"][3][0,3], 
                robot.gripper.info["tcp"][3][1,3], 
                robot.gripper.info["tcp"][3][2,3], s=5, c='r')