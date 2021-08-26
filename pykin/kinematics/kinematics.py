import numpy as np
from collections import OrderedDict

from pykin.kinematics import jacobian as jac
from pykin.utils import transform_utils as tf
from pykin.utils.kin_utils import Baxter, calc_pose_error, convert_thetas_to_dict, logging_time

class Kinematics:
    def __init__(self, 
                robot_name, 
                offset, 
                active_joint_names=[],
                base_name="base", 
                eef_name=None, 
                frames=None,
                ):
        self.robot_name = robot_name
        self.offset = offset
        self.active_joint_names = active_joint_names
        self.base_name = base_name
        self.eef_name = eef_name
        self.frames = frames
        self._transformations = None

    @property
    def offset(self):
        return self._offset
    
    @offset.setter
    def offset(self, offset):
        self._offset = offset

    @property
    def active_joint_names(self):
        return self._active_joint_names
    
    @active_joint_names.setter
    def active_joint_names(self, active_joint_names):
        self._active_joint_names = active_joint_names

    @property
    def base_name(self):
        return self._base_name
    
    @base_name.setter
    def base_name(self, base_name):
        self._base_name = base_name

    @property
    def eef_name(self):
        return self._eef_name
    
    @eef_name.setter
    def eef_name(self, eef_name):
        self._eef_name = eef_name

    @property
    def frames(self):
        return self._frames
    
    @frames.setter
    def frames(self, frames):
        self._frames = frames

    def forward_kinematics(self, thetas, collision_check=False):
        if not isinstance(self.frames, list):
            thetas = convert_thetas_to_dict(self.active_joint_names, thetas)
        self._transformations = self._compute_FK(self.frames, self.offset, thetas)
        return self._transformations
    
    @logging_time
    def inverse_kinematics(self, current_joints, target_pose, method="LM", maxIter=1000):
        if method == "NR":
            joints, trajectory_joints = self._compute_IK_NR(
                current_joints, 
                target_pose, 
                maxIter=maxIter
            )
        if method == "LM":
            joints, trajectory_joints = self._compute_IK_LM(
                current_joints, 
                target_pose, 
                maxIter=maxIter
            )
        return joints, trajectory_joints

    def _compute_FK(self, frames, offset, thetas):
        transformations = OrderedDict()
        if isinstance(thetas, dict):
            trans = offset * frames.get_transform(thetas.get(frames.joint.name, 0.0))
            transformations[frames.link.name] = trans * frames.link.offset
            for child in frames.children:
                transformations.update(self._compute_FK(child, trans, thetas))
        else:
            cnt = 0
            trans = offset
            for frame in frames:
                trans = trans * frame.get_transform(thetas[cnt])
                transformations[frame.link.name] = trans * frame.link.offset
                if frame.joint.dtype != "fixed":
                    cnt += 1
                if cnt >= len(thetas):
                    cnt -= 1     
                if self.robot_name == "baxter":
                    Baxter.add_visual_link(transformations, frame)
        return transformations

    def _compute_IK_NR(self, current_joints, target_pose, maxIter):
        lamb = 0.5
        iterator = 0
        EPS = float(1e-6)
        dof = len(current_joints)

        # Step 1. Prepare the position and attitude of the target link
        target_pose = tf.get_homogeneous_matrix(target_pose[:3], target_pose[3:])

        # Step 2. Use forward kinematics to calculate the position and attitude of the target link
        cur_fk = self.forward_kinematics(current_joints)
        cur_pose = list(cur_fk.values())[-1].matrix()

        # Step 3. Calculate the difference in position and attitude
        err_pose = calc_pose_error(target_pose, cur_pose, EPS)
        err = np.linalg.norm(err_pose)

        # Step 4. If error is small enough, stop the calculation
        trajectory_joints = []
        while err > EPS:
            # Avoid infinite calculation
            iterator += 1
            if iterator > maxIter:
                break
            
            # Step 5. If error is not small enough, calculate dq which would reduce the error 
            # Get jacobian to calculate dq 
            J = jac.calc_jacobian(self.frames, cur_fk, current_joints)
            dq = lamb * np.dot(np.linalg.pinv(J), err_pose)

            # Step 6. Update joint angles by q = q + dq and calculate forward Kinematics
            current_joints = [current_joints[i] + dq[i] for i in range(dof)]
            trajectory_joints.append(np.array([float(current_joint) for current_joint in current_joints]))
            cur_fk = self.forward_kinematics(current_joints)

            cur_pose = list(cur_fk.values())[-1].matrix()
            err_pose = calc_pose_error(target_pose, cur_pose, EPS)
            err = np.linalg.norm(err_pose)

        print(f"Iterators : {iterator-1}")
        current_joints = np.array([float(current_joint) for current_joint in current_joints])
        return current_joints, trajectory_joints

    def _compute_IK_LM(self, current_joints, target, maxIter):
        iterator = 0
        EPS = float(1E-12)
        dof = len(current_joints)
        wn_pos = 1/0.3
        wn_ang = 1/(2*np.pi)
        We = np.diag([wn_pos, wn_pos, wn_pos, wn_ang, wn_ang, wn_ang])
        Wn = np.eye(dof)

        # Step 1. Prepare the position and attitude of the target link
        target_pose = tf.get_homogeneous_matrix(target[:3], target[3:])

        # Step 2. Use forward kinematics to calculate the position and attitude of the target link
        cur_fk = self.forward_kinematics(current_joints)
        cur_pose = list(cur_fk.values())[-1].matrix()

        # # Step 3. Calculate the difference in position and attitude
        err = calc_pose_error(target_pose, cur_pose, EPS)
        Ek = float(np.dot(np.dot(err.T, We), err)[0])

        trajectory_joints = []
        # # Step 4. If error is small enough, stop the calculation
        while Ek > EPS:
            # Avoid infinite calculation
            iterator += 1
            if iterator > maxIter:
                break
            
            lamb = Ek + 0.002

            # Step 5. If error is not small enough, calculate dq which would reduce the error
            # Get jacobian to calculate dq
            J = jac.calc_jacobian(self.frames, cur_fk, current_joints)
            Jh = np.dot(np.dot(J.T, We), J) + np.dot(Wn, lamb)
            
            gerr = np.dot(np.dot(J.T, We), err)
            dq = np.dot(np.linalg.pinv(Jh), gerr)

            # Step 6. Update joint angles by q = q + dq and calculate forward Kinematics
            current_joints = [current_joints[i] + dq[i] for i in range(dof)]
            trajectory_joints.append(np.array([float(current_joint) for current_joint in current_joints]))
            
            cur_fk = self.forward_kinematics(current_joints)
            cur_pose = list(cur_fk.values())[-1].matrix()
            err = calc_pose_error(target_pose, cur_pose, EPS)
            Ek2 = float(np.dot(np.dot(err.T, We), err)[0])
            
            if Ek2 < Ek:
                Ek = Ek2
            else:
                current_joints = [current_joints[i] - dq[i] for i in range(dof)]
                cur_fk = self.forward_kinematics(current_joints)
                break
            
        print(f"Iterators : {iterator-1}")
        current_joints = np.array([float(current_joint) for current_joint in current_joints])
        return current_joints, trajectory_joints
