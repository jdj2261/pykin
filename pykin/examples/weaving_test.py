import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from pykin.utils import plot_utils as p_utils
from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.utils import transform_utils as t_utils

# 로봇 모델 로드 (예: Doosan 로봇)
file_path = "urdf/doosan/doosan_with_robotiq140.urdf"
robot = SingleArm(
    file_path,
    Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]),
    has_gripper=True,
    gripper_name="robotiq140",
)
robot.setup_link_name("base_0", "right_hand")


def calculate_weaving_position(
    t,
    start_pos,
    end_pos,
    freq,
    left_dist,
    right_dist,
    angle,
    wall_direction,
    offset_angle,
    progress_angle,
    boundary_limit,
):
    # 용접선 방향 벡터
    weld_direction = end_pos - start_pos
    weld_length = np.linalg.norm(weld_direction)
    weld_unit = weld_direction / weld_length

    # 진행 거리 계산
    progress = t * weld_length

    # 끝에 도달했는지 확인
    if progress >= weld_length:
        return end_pos, True

    # 기본 위빙 패턴 (삼각형)
    lateral_offset = (left_dist + right_dist) / 2 * np.sin(2 * np.pi * freq * t)
    vertical_offset = -(left_dist + right_dist) / 4 * np.sin(4 * np.pi * freq * t)

    # 벽 방향 벡터 계산
    if wall_direction == "vertical":
        wall_vector = np.array([0, 0, 1])
    elif wall_direction == "horizontal":
        wall_vector = np.cross(weld_unit, [0, 0, 1])
    else:  # torch_based
        # 토치 자세 기준 (여기서는 간단히 구현)
        wall_vector = np.cross(weld_unit, [0, 1, 0])

    # 옵셋 각도 적용
    rotation_matrix = t_utils.get_rotation_matrix([0, 0, offset_angle])
    wall_vector = rotation_matrix.dot(wall_vector)

    # 위빙 방향 벡터 계산
    lateral_direction = np.cross(weld_unit, wall_vector)
    lateral_direction /= np.linalg.norm(lateral_direction)

    # 진행 각도 적용
    progress_rotation = t_utils.get_rotation_matrix([0, 0, progress_angle])
    lateral_direction = progress_rotation.dot(lateral_direction)

    # 수직 방향 벡터
    vertical_direction = np.cross(weld_unit, lateral_direction)

    # 위치 계산
    pos = (
        start_pos
        + progress * weld_unit
        + lateral_offset * lateral_direction
        + vertical_offset * vertical_direction
    )

    # 경계 제한 적용
    if boundary_limit == "유효":
        proj_start = np.dot(pos - start_pos, weld_unit)
        proj_end = np.dot(pos - end_pos, weld_unit)
        if proj_start < 0:
            pos = (
                start_pos
                + lateral_offset * lateral_direction
                + vertical_offset * vertical_direction
            )
        elif proj_end > 0:
            pos = (
                end_pos + lateral_offset * lateral_direction + vertical_offset * vertical_direction
            )

    return pos, False


# 두 부재 생성 및 용접선 정의
def create_parts_and_weld_line():
    part1 = np.array([[1.4, -0.1, 0.6], [1.6, -0.1, 0.6], [1.6, 0.3, 0.6], [1.4, 0.3, 0.6]])
    part2 = np.array([[1.5, -0.1, 0.6], [1.5, -0.1, 0.8], [1.5, 0.3, 0.8], [1.5, 0.3, 0.6]])
    weld_start = np.array([1.5, -0.1, 0.6])
    weld_end = np.array([1.5, 0.3, 0.6])
    return part1, part2, weld_start, weld_end


part1, part2, weld_start, weld_end = create_parts_and_weld_line()

# 위빙 파라미터 설정
freq = 2.0  # 위빙 주파수
left_dist = 0.03  # 좌 방향 거리
right_dist = 0.03  # 우 방향 거리
angle = np.radians(90)  # 위빙 각도
wall_direction = "vertical"  # 벽 방향 ("vertical", "horizontal", "torch_based")
offset_angle = np.radians(0)  # 옵셋 각도
progress_angle = np.radians(0)  # 진행 각도 (용접선 방향으로 설정)
boundary_limit = "유효"  # 경계 제한 ("유효" 또는 "무효")
duration = 5.0  # 총 위빙 시간
steps = 100  # 시뮬레이션 스텝 수

# 위빙 궤적 생성 및 IK 계산
trajectory = []
joint_angles = []
t = 0
dt = duration / steps

for i in range(steps):
    pos, is_end = calculate_weaving_position(
        t,
        weld_start,
        weld_end,
        freq,
        left_dist,
        right_dist,
        angle,
        wall_direction,
        offset_angle,
        progress_angle,
        boundary_limit,
    )

    # 용접 방향을 향하도록 방향 설정
    weld_direction = weld_end - weld_start
    weld_direction /= np.linalg.norm(weld_direction)
    z_axis = -weld_direction  # 용접 방향의 반대 방향을 z축으로

    y_axis = np.cross([0, 0, 1], z_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)

    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    target_pose = np.eye(4)
    target_pose[:3, :3] = rotation_matrix
    target_pose[:3, 3] = pos

    joint_angle = robot.inverse_kin(robot.init_qpos, target_pose)
    trajectory.append(pos)
    joint_angles.append(joint_angle)

    if is_end:
        break

    t += dt


trajectory = np.array(trajectory)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# 애니메이션 업데이트 함수
def update(frame):
    ax.clear()
    ax.set_xlim([1.3, 1.7])
    ax.set_ylim([-0.2, 0.4])
    ax.set_zlim([0.5, 0.9])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Weaving Animation (Frame {frame+1}/{steps})")

    # 두 부재 그리기
    ax.add_collection3d(Poly3DCollection([part1], alpha=0.5, color="blue"))
    ax.add_collection3d(Poly3DCollection([part2], alpha=0.5, color="green"))

    # 용접선 그리기
    ax.plot(
        [weld_start[0], weld_end[0]],
        [weld_start[1], weld_end[1]],
        [weld_start[2], weld_end[2]],
        "y-",
        linewidth=2,
    )

    # 위빙 궤적 그리기
    ax.plot(
        trajectory[: frame + 1, 0],
        trajectory[: frame + 1, 1],
        trajectory[: frame + 1, 2],
        "r-",
        linewidth=2,
    )
    ax.scatter(trajectory[frame, 0], trajectory[frame, 1], trajectory[frame, 2], c="red", s=50)

    # 로봇 그리기
    if frame < len(joint_angles):
        robot.set_transform(joint_angles[frame])
        p_utils.plot_robot(ax=ax, robot=robot, geom="visual", only_visible_geom=False)

    return ax


# 애니메이션 생성
anim = FuncAnimation(fig, update, frames=np.arange(len(trajectory)), interval=50, blit=False)
plt.show()

# 애니메이션 저장 (선택사항)
# anim.save('weaving_animation.gif', writer='pillow', fps=30)
