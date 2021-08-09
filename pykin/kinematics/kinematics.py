import numpy as np
from collections import OrderedDict

from pykin.kinematics.transform import Transform
from pykin.kinematics import transformation as tf
from pykin.kinematics import jacobian as jac


class Kinematics:
    def __init__(self, tree):
        self.tree = tree

    @staticmethod
    def _forward_kinematics(root, theta_dict, offset=Transform()):
        link_transforms = OrderedDict()
        trans = offset * root.get_transform(theta_dict.get(root.joint.name, 0.0))
        link_transforms[root.link.name] = trans * root.link.offset
        for child in root.children:
            link_transforms.update(
                Kinematics._forward_kinematics(child, theta_dict, trans)
            )

        return link_transforms

    def forward_kinematics(self, thetas, offset=Transform(), desired_tree=None):
        if desired_tree is None:
            if not isinstance(thetas, dict):
                joint_names = self.tree.get_joint_parameter_names
                assert len(joint_names) == len(
                    thetas
                ), f"the number of joints is {len(joint_names)}, but the number of joint's angle is {len(thetas)}"
                thetas_dict = dict((j, thetas[i]) for i, j in enumerate(joint_names))
            else:
                thetas_dict = thetas
            return self._forward_kinematics(self.tree.root, thetas_dict, offset)
        else:
            cnt = 0
            link_transforms = {}
            trans = offset
            for f in desired_tree:
                trans = trans * f.get_transform(thetas[cnt])
                link_transforms[f.link.name] = trans * f.link.offset
                if f.joint.dtype != "fixed":
                    cnt += 1
                if cnt >= len(thetas):
                    cnt -= 1
            return link_transforms

    def analytical_inverse_kinematics(self, pose):
        # Link Length [m]
        L = [0.27035, 0.069, 0.36435, 0.069, 0.37429, 0.0, 0.38735]  # 0.36830
        L_h = 0.37082  # np.sqrt(L[2] ** 2 + L[3] ** 2)


        T_BL_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, L[0]], [0, 0, 0, 1]])
        T_BR_0 = T_BL_0

        TM_6_GL = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, L[6]], [0, 0, 0, 1]])
        TM_6_GR = TM_6_GL

        # pose to Transformation matrix
        T_BL_Goal = tf.pose_to_homogeneous(pose)

        T_0_6 = np.dot(np.dot(np.linalg.inv(T_BL_0), T_BL_Goal), np.linalg.inv(TM_6_GL))

        # position
        x = T_0_6[0, 3]
        y = T_0_6[1, 3]
        z = T_0_6[2, 3]

        # calculate theta 1
        theta_1 = np.arctan2(y, x)

        # Parameters for calculating theta 2
        E = 2 * L_h * (L[1] - (x / np.cos(theta_1)))
        F = 2 * L_h * z
        G = (
            ((x ** 2) / (np.cos(theta_1) ** 2))
            + (L[1] ** 2 + L_h ** 2 - L[4] ** 2)
            + (z ** 2)
            - ((2 * L[1] * x) / (np.cos(theta_1)))
        )

        k = E ** 2 + F ** 2 - G ** 2
        print(k)
        if k > 0:
            k = np.sqrt(k)
        else:
            print("k is lower 0")
            k = 0

        # tangent 1, 2 for calculating theta 2
        tan_1 = (-F + k) / (G - E)
        tan_2 = (-F - k) / (G - E)

        # calcuate theta_2_1, theta_2_2
        theta_2_1 = 2 * np.arctan(tan_1)
        theta_2_2 = 2 * np.arctan(tan_2)

        theta_3 = 0

        # calcuate theta 4_1, 4_2
        theta_4_1 = (
            np.arctan2(
                (-z - L_h * np.sin(theta_2_1)),
                ((x / np.cos(theta_1) - L[1] - L_h * np.cos(theta_2_1))),
            )
            - theta_2_1
        )
        theta_4_2 = (
            np.arctan2(
                (-z - L_h * np.sin(theta_2_2)),
                ((x / np.cos(theta_1) - L[1] - L_h * np.cos(theta_2_2))),
            )
            - theta_2_2
        )
        R_0_6 = T_0_6[0:3, 0:3]

        theta_2_4_1 = theta_2_1 + theta_4_1
        R_0_3_1 = np.array(
            [
                [
                    -np.cos(theta_1) * np.sin(theta_2_4_1),
                    -np.cos(theta_1) * np.cos(theta_2_4_1),
                    -np.sin(theta_1),
                ],
                [
                    -np.sin(theta_1) * np.sin(theta_2_4_1),
                    -np.sin(theta_1) * np.cos(theta_2_4_1),
                    np.cos(theta_1),
                ],
                [-np.cos(theta_2_4_1), np.sin(theta_2_4_1), 0],
            ]
        )
        R_3_6_1 = np.dot(np.matrix.transpose(R_0_3_1), R_0_6)

        theta_2_4_2 = theta_2_2 + theta_4_2
        R_0_3_2 = np.array(
            [
                [
                    -np.cos(theta_1) * np.sin(theta_2_4_2),
                    -np.cos(theta_1) * np.cos(theta_2_4_2),
                    -np.sin(theta_1),
                ],
                [
                    -np.sin(theta_1) * np.sin(theta_2_4_2),
                    -np.sin(theta_1) * np.cos(theta_2_4_2),
                    np.cos(theta_1),
                ],
                [-np.cos(theta_2_4_2), np.sin(theta_2_4_2), 0],
            ]
        )
        R_3_6_2 = np.dot(np.matrix.transpose(R_0_3_2), R_0_6)

        theta_5_1 = np.arctan2(R_3_6_1[2, 2], R_3_6_1[0, 2])
        theta_5_2 = np.arctan2(R_3_6_2[2, 2], R_3_6_2[0, 2])

        theta_7_1 = np.arctan2(-R_3_6_1[1, 1], R_3_6_1[1, 0])
        theta_7_2 = np.arctan2(-R_3_6_2[1, 1], R_3_6_2[1, 0])

        theta_6_1 = np.arctan2((R_3_6_1[1, 0] / np.cos(theta_7_1)), -R_3_6_1[1, 2])
        theta_6_2 = np.arctan2((R_3_6_2[1, 0] / np.cos(theta_7_2)), -R_3_6_2[1, 2])

        result1 = [
            theta_1,
            theta_2_1,
            theta_3,
            theta_4_1,
            theta_5_1,
            theta_6_1,
            theta_7_1,
        ]
        result2 = [
            theta_1,
            theta_2_2,
            theta_3,
            theta_4_2,
            theta_5_2,
            theta_6_2,
            theta_7_2,
        ]

        return result1, result2

    # TODO
    # singularity problem
    # Initial Joints Random pick
    # Trajectory 
    # self collision checker 
    # joint limit


    def calc_pose_error(self, T_ref, T_cur, EPS):

        def rot_to_omega(R):
            # referred p36
            el = np.array(
                [[R[2, 1] - R[1, 2]],
                    [R[0, 2] - R[2, 0]],
                    [R[1, 0] - R[0, 1]]]
            )
            norm_el = np.linalg.norm(el)
            if norm_el > EPS:
                w = np.dot(np.arctan2(norm_el, np.trace(R) - 1) / norm_el, el)
            elif (R[0, 0] > 0 and R[1, 1] > 0 and R[2, 2] > 0):
                w = np.zeros((3, 1))
            else:
                w = np.dot(
                    np.pi/2, np.array([[R[0, 0] + 1], [R[1, 1] + 1], [R[2, 2] + 1]]))
            return w

        pos_err = np.array([T_ref[:3, -1] - T_cur[:3, -1]])
        rot_err = np.dot(T_cur[:3, :3].T, T_ref[:3, :3])
        w_err = np.dot(T_cur[:3, :3], rot_to_omega(rot_err))

        return np.vstack((pos_err.T, w_err))

    def numerical_inverse_kinematics_NR(self, current_joints, target, desired_tree, lower, upper, maxIter):

        lamb = 0.5
        iterator = 0
        maxIter = maxIter
        EPS = float(1e-6)
        dof = len(current_joints)


        # Step 1. Prepare the position and attitude of the target link
        target_pose = tf.get_homogeneous_matrix(target[:3], target[3:])

        # Step 2. Use forward kinematics to calculate the position and attitude of the target link
        cur_fk = self.forward_kinematics(current_joints, desired_tree=desired_tree)
        cur_pose = list(cur_fk.values())[-1].matrix()

        # Step 3. Calculate the difference in position and attitude
        err_pose = self.calc_pose_error(target_pose, cur_pose, EPS)
        err = np.linalg.norm(err_pose)
        # out = [0 for i in range(dof)]
        # Step 4. If error is small enough, stop the calculation
        while err > EPS:
            # Step 5. If error is not small enough, calculate dq which would reduce the error 
            # Get jacobian to calculate dq 
            J = jac.calc_jacobian(desired_tree, cur_fk, current_joints)
            dq = lamb * np.dot(np.linalg.pinv(J), err_pose)

            # Step 6. Update joint angles by q = q + dq and calculate forward Kinematics
            current_joints = [current_joints[i] + dq[i] for i in range(dof)]

            if lower is not None and upper is not None:
                for i in range(len(current_joints)):
                    if current_joints[i] < lower[i]:
                        current_joints[i] = lower[i]
                    if current_joints[i] > upper[i]:
                        current_joints[i] = upper[i]

            cur_fk = self.forward_kinematics(
                current_joints, desired_tree=desired_tree)
            cur_pose = list(cur_fk.values())[-1].matrix()
            err_pose = self.calc_pose_error(target_pose, cur_pose, EPS)
            err = np.linalg.norm(err_pose)

            # Avoid infinite calculation
            iterator += 1
            if iterator > maxIter:
                break

        print(f"Iterators : {iterator}")
        current_joints = np.array([float(current_joint)
                                  for current_joint in current_joints])
        return current_joints

    def numerical_inverse_kinematics_LM(self, current_joints, target, desired_tree, lower, upper, maxIter):

        iterator = 0
        maxIter = maxIter
        EPS = float(1E-12)
        dof = len(current_joints)
        wn_pos = 1/0.3
        wn_ang = 1/(2*np.pi)
        We = np.diag([wn_pos, wn_pos, wn_pos, wn_ang, wn_ang, wn_ang])
        Wn = np.eye(dof)

        # Step 1. Prepare the position and attitude of the target link
        target_pose = tf.get_homogeneous_matrix(target[:3], target[3:])

        # Step 2. Use forward kinematics to calculate the position and attitude of the target link
        cur_fk = self.forward_kinematics(
            current_joints, desired_tree=desired_tree)
        cur_pose = list(cur_fk.values())[-1].matrix()

        # # Step 3. Calculate the difference in position and attitude
        err = self.calc_pose_error(target_pose, cur_pose, EPS)
        Ek = float(np.dot(np.dot(err.T, We), err)[0])

        # # Step 4. If error is small enough, stop the calculation
        while Ek > EPS:
            # Avoid infinite calculation
            iterator += 1
            if iterator > maxIter:
                break
            
            lamb = Ek + 0.002
            # Step 5. If error is not small enough, calculate dq which would reduce the error
            # Get jacobian to calculate dq
            J = jac.calc_jacobian(desired_tree, cur_fk, current_joints)
            Jh = np.dot(np.dot(J.T, We), J) + np.dot(Wn, lamb)
            
            gerr = np.dot(np.dot(J.T, We), err)
            dq = np.dot(np.linalg.pinv(Jh), gerr)

            # Step 6. Update joint angles by q = q + dq and calculate forward Kinematics
            current_joints = [current_joints[i] + dq[i] for i in range(dof)]

            if lower is not None and upper is not None:
                for i in range(len(current_joints)):
                    if current_joints[i] < lower[i]:
                        current_joints[i] = lower[i]
                    if current_joints[i] > upper[i]:
                        current_joints[i] = upper[i]

            cur_fk = self.forward_kinematics(
                current_joints, desired_tree=desired_tree)
            cur_pose = list(cur_fk.values())[-1].matrix()
            err = self.calc_pose_error(target_pose, cur_pose, EPS)
            Ek2 = float(np.dot(np.dot(err.T, We), err)[0])
            if Ek2 < Ek:
                Ek = Ek2
            else:
                current_joints = [current_joints[i] - dq[i]
                                  for i in range(dof)]
                cur_fk = self.forward_kinematics(
                    current_joints, desired_tree=desired_tree)
                break
        print(f"Iterators : {iterator}")
        current_joints = np.array([float(current_joint)
                                  for current_joint in current_joints])
        return current_joints
