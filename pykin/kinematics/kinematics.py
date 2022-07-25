import numpy as np
from collections import OrderedDict

from pykin.kinematics import jacobian as jac
from pykin.kinematics.transform import Transform
from pykin.utils import transform_utils as t_utils
from pykin.utils.kin_utils import calc_pose_error, convert_thetas_to_dict, logging_time


class Kinematics:
    """
    Class of Kinematics

    Args:
        robot_name (str): robot's name
        offset (Transform): robot's offset
        active_joint_names (list): robot's actuated joints
        base_name (str): reference link's name
        eef_name (str): end effector's name
    """
    def __init__(self, 
                robot_name, 
                offset, 
                active_joint_names=[],
                base_name="base", 
                eef_name=None, 
                ):
        self.robot_name = robot_name
        self.offset = offset
        self.active_joint_names = active_joint_names
        self.base_name = base_name
        self.eef_name = eef_name

    def forward_kinematics(self, frames, thetas):
        """
        Returns transformations obtained by computing fk

        Args:
            frames (list or Frame()): robot's frame for forward kinematics
            thetas (sequence of float): input joint angles

        Returns:
            fk (OrderedDict): transformations
        """

        if not isinstance(frames, list) :
            thetas = convert_thetas_to_dict(self.active_joint_names, thetas)
        fk = self._compute_FK(frames, self.offset, thetas)
        return fk
    
    @logging_time
    def inverse_kinematics(
        self, 
        frames, 
        current_joints, 
        target_pose, 
        method="LM", 
        max_iter=1000
    ):
        """
        Returns joint angles obtained by computing IK
        
        Args:
            frames (Frame()): robot's frame for invers kinematics
            current_joints (sequence of float): input joint angles
            target_pose (np.array): goal pose to achieve
            method (str): two methods to calculate IK (LM: Levenberg-marquardt, NR: Newton-raphson)
            max_iter (int): Maximum number of calculation iterations

        Returns:
            joints (np.array): target joint angles
        """
        if method == "NR":
            joints = self._compute_IK_NR(
                frames,
                current_joints, 
                target_pose, 
                max_iter=max_iter
            )
        if method == "LM":
            joints = self._compute_IK_LM(
                frames,
                current_joints, 
                target_pose, 
                max_iter=max_iter
            )
        if method == "LM2":
            joints = self._compute_IK_LM2(
                frames,
                current_joints, 
                target_pose, 
                max_iter=max_iter
            )
        if method == "GaBO":
            joints = self._compute_IK_GaBO(
                frames,
                target_pose,
                max_iter=max_iter,
                opt_dimension=2
            )
        return joints

    def _compute_FK(self, frames, offset, thetas):
        """
        Computes forward kinematics

        Args:
            frames (list or Frame()): robot's frame for forward kinematics
            offset (Transform): robot's offset
            thetas (sequence of float): input joint angles

        Returns:
            fk (OrderedDict): transformations
        """
        fk = OrderedDict()
        if not isinstance(frames, list):
            trans = offset * frames.get_transform(thetas.get(frames.joint.name, 0.0))
            fk[frames.link.name] = trans
            for child in frames.children:
                fk.update(self._compute_FK(child, trans, thetas))
        else:
            # To compute IK
            cnt = 0
            trans = offset
            for frame in frames:
                trans = trans * frame.get_transform(thetas[cnt])
                fk[frame.link.name] = trans
                
                if frame.joint.dtype != "fixed":
                    cnt += 1
                
                if cnt >= len(thetas):
                    cnt -= 1     
                
                if self.robot_name == "baxter":
                    Baxter.add_visual_link(fk, frame)

        return fk

    def _compute_IK_NR(
        self, 
        frames, 
        current_joints, 
        target_pose, 
        max_iter
    ):
        """
        Computes inverse kinematics using Newton Raphson method

        Args:
            frames (list or Frame()): robot's frame for inverse kinematics
            current_joints (sequence of float): input joint angles
            target_pose (np.array): goal pose to achieve
            max_iter (int): Maximum number of calculation iterations

        Returns:
            joints (np.array): target joint angles
        """
        lamb = 0.5
        iterator = 1
        EPS = float(1e-6)
        dof = len(current_joints)

        target_pose = t_utils.get_h_mat(target_pose[:3], target_pose[3:])

        cur_fk = self.forward_kinematics(frames, current_joints)
        cur_pose = list(cur_fk.values())[-1].h_mat

        err_pose = calc_pose_error(target_pose, cur_pose, EPS)
        err = np.linalg.norm(err_pose)

        while err > EPS:

            iterator += 1
            if iterator > max_iter:
                break
            
            J = jac.calc_jacobian(frames, cur_fk, len(current_joints))
            dq = lamb * np.dot(np.linalg.pinv(J), err_pose)
            current_joints = [current_joints[i] + dq[i] for i in range(dof)]
            cur_fk = self.forward_kinematics(frames, current_joints)

            cur_pose = list(cur_fk.values())[-1].h_mat
            err_pose = calc_pose_error(target_pose, cur_pose, EPS)
            err = np.linalg.norm(err_pose)

        print(f"Iterators : {iterator-1}")
        current_joints = np.array([float(current_joint) for current_joint in current_joints])
        return current_joints

    def _compute_IK_LM(
        self, 
        frames, 
        current_joints, 
        target_pose, 
        max_iter
    ):
        """
        Computes inverse kinematics using Levenberg-Marquatdt method

        Args:
            frames (list or Frame()): robot's frame for inverse kinematics
            current_joints (sequence of float): input joint angles
            target_pose (np.array): goal pose to achieve
            max_iter (int): Maximum number of calculation iterations

        Returns:
            joints (np.array): target joint angles
        """
        iterator = 1
        EPS = float(1E-12)
        dof = len(current_joints)
        wn_pos = 1/0.3
        wn_ang = 1/(2*np.pi)
        We = np.diag([wn_pos, wn_pos, wn_pos, wn_ang, wn_ang, wn_ang])
        Wn = np.eye(dof)

        target_pose = t_utils.get_h_mat(target_pose[:3], target_pose[3:])

        cur_fk = self.forward_kinematics(frames, current_joints)
        cur_pose = list(cur_fk.values())[-1].h_mat

        err = calc_pose_error(target_pose, cur_pose, EPS)
        Ek = float(np.dot(np.dot(err.T, We), err)[0])

        while Ek > EPS:
            iterator += 1
            if iterator > max_iter:
                break
            
            lamb = Ek + 0.002

            J = jac.calc_jacobian(frames, cur_fk, len(current_joints))
            J_dls = np.dot(np.dot(J.T, We), J) + np.dot(Wn, lamb)
            
            gerr = np.dot(np.dot(J.T, We), err)
            dq = np.dot(np.linalg.inv(J_dls), gerr)
            current_joints = [current_joints[i] + dq[i] for i in range(dof)]
           
            cur_fk = self.forward_kinematics(frames, current_joints)
            cur_pose = list(cur_fk.values())[-1].h_mat
            err = calc_pose_error(target_pose, cur_pose, EPS)
            Ek2 = float(np.dot(np.dot(err.T, We), err)[0])
            
            if Ek2 < Ek:
                Ek = Ek2
            else:
                current_joints = [current_joints[i] - dq[i] for i in range(dof)]
                cur_fk = self.forward_kinematics(frames, current_joints)
                break
            
        print(f"Iterators : {iterator-1}")
        current_joints = np.array([float(current_joint) for current_joint in current_joints])
        return current_joints

    def _compute_IK_LM2(
        self, 
        frames, 
        current_joints, 
        target_pose, 
        max_iter
    ):
        """
        Computes inverse kinematics using Levenberg-Marquatdt method

        Args:
            frames (list or Frame()): robot's frame for inverse kinematics
            current_joints (sequence of float): input joint angles
            target_pose (np.array): goal pose to achieve
            max_iter (int): Maximum number of calculation iterations

        Returns:
            joints (np.array): target joint angles
        """
        iterator = 1
        EPS = float(1E-12)
        dof = len(current_joints)
        wn_pos = 1/0.3
        wn_ang = 1/(2*np.pi)
        We = np.diag([wn_pos, wn_pos, wn_pos, wn_ang, wn_ang, wn_ang])
        Wn = np.eye(dof)

        target_pose = t_utils.get_h_mat(target_pose[:3], target_pose[3:])

        cur_fk = self.forward_kinematics(frames, current_joints)
        cur_pose = list(cur_fk.values())[-1].h_mat

        err = calc_pose_error(target_pose, cur_pose, EPS)
        Ek = float(np.dot(np.dot(err.T, We), err)[0])

        while Ek > EPS:
            iterator += 1
            if iterator > max_iter:
                break
            
            lamb = Ek + 0.002

            J = jac.calc_jacobian(frames, cur_fk, len(current_joints))

            JT = np.dot(np.dot(J.T, We), J)
            J_dls = JT + np.dot(np.diag(np.diag(JT)), lamb)
            
            gerr = np.dot(np.dot(J.T, We), err)
            dq = np.dot(np.linalg.inv(J_dls), gerr)
            current_joints = [current_joints[i] + dq[i] for i in range(dof)]
           
            cur_fk = self.forward_kinematics(frames, current_joints)
            cur_pose = list(cur_fk.values())[-1].h_mat
            err = calc_pose_error(target_pose, cur_pose, EPS)
            Ek2 = float(np.dot(np.dot(err.T, We), err)[0])
            
            if Ek2 < Ek:
                Ek = Ek2
            else:
                current_joints = [current_joints[i] - dq[i] for i in range(dof)]
                cur_fk = self.forward_kinematics(frames, current_joints)
                break
            
        print(f"Iterators : {iterator-1}")
        current_joints = np.array([float(current_joint) for current_joint in current_joints])
        return current_joints
    
    def _compute_IK_GaBO(
        self,
        frames,
        target_pose,
        max_iter,
        opt_dimension,
    ):
        """
        Computes inverse kinematics using Geometric-aware Bayesian Optimization method

        Args:
            frames (list or Frame()): robot's frame for forward kinematics
            target_pose (np.array): goal pose to achieve
            max_iter (int): Maximum number of bayesian optimization iterations
            opt_dimension (int) : torus dimension to optimize from end-effector frame to backward order (Recommended : 2~3)

        Returns:
            joints (np.array): target joint angles
        """
        try:
            import torch, gpytorch, botorch, pymanopt
        except ImportError:
            import os
            print("ImportError: No module found to run GaBO IK method. Try install requirements")
            os.system("pip install -r pykin/utils/gabo/requirements.txt")
            print("Requirement installation finished. Please re-execute python file")
            os._exit(0)
        from botorch.acquisition import ExpectedImprovement
        from pykin.utils.gabo.module.torus import Torus
        from pykin.utils.gabo.module.manifold_optimize import joint_optimize_manifold
        from pykin.utils.error_utils import BimanualTypeError
        import pykin.utils.gabo.gabo_util as g_util
        
        if self.robot_name == "baxter":
            raise BimanualTypeError

        # Check cuda device, torch setting
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
        else:
            device = 'cpu'
        torch.set_default_dtype(torch.float32)

        # Define pose error objective function
        def get_pose_error(target_pose, cur_point):
            cur_angle = g_util.convert_point_to_angle_torch(cur_point)
            cur_fk = self.forward_kinematics(frames, cur_angle)
            cur_pose = list(cur_fk.values())[-1].h_mat
            err = calc_pose_error(target_pose, cur_pose, EPS)
            return np.linalg.norm(err)

        target_pose = t_utils.get_h_mat(target_pose[:3], target_pose[3:])

        # Define hyperparameters
        opt_dimension = opt_dimension
        robot_dimension = len(self.active_joint_names)
        nb_data_init = 10000
        nb_iter_bo = max_iter
        enough_sample = 4
        EPS = float(1e-6)


        # Define torus manifold
        opt_manifold = Torus(dimension=opt_dimension)
        robot_manifold = Torus(dimension=robot_dimension)
        
        # Get initial x data batches
        print("Start Pose Random Sampling")
        scaled_x = []   
        scaled_y = []
        
        while True:
            x_init = np.array([np.array(robot_manifold.rand()).reshape(2 * robot_dimension) for n in range(nb_data_init)])
            x_data = torch.tensor(x_init)
            y_data = torch.zeros(nb_data_init, dtype=torch.float64)

            for n in range(nb_data_init):
                err = get_pose_error(target_pose, cur_point=x_data[n])
                y_data[n] = err
            
            for idx, data in enumerate(y_data):
                if data < 0.4:
                    scaled_y.append(y_data[idx])
                    scaled_x.append(x_data[idx])
            
            if len(scaled_y) > enough_sample:
                break
            print(f"\tNot enough samples.. Resampling ({len(scaled_y)} collected)")

        x_data = torch.stack(scaled_x, 0)
        y_data = torch.stack(scaled_y, 0)
        reduced_x_data = x_data[:, -opt_dimension * 2:]
        print(f"Sampling Done : Collected proper samples {y_data.shape[0]}")

        # Initialize best observation and function value list
        new_best_f, index = y_data.min(0)
        best_x = [x_data[index]]
        best_f = [new_best_f]

        # Adjust x, y data to new data format
        determined_joint = best_x[0][:-opt_dimension*2]
        x_data = reduced_x_data

        for n in range(y_data.shape[0]):
            adjusted_x = torch.cat([determined_joint, x_data[n]])
            err = get_pose_error(target_pose, cur_point=adjusted_x)
            y_data[n] = err

        # Re-calculate best observation
        new_best_f, index = y_data.min(0)
        best_x = [x_data[index]]    
        best_f = [new_best_f]
        print(f"Initial best guess of error {best_f[0]}")

        determined_joint = determined_joint.to(device)
        x_data = x_data.to(device)
        y_data = y_data.to(device)

        # Define the GPR model
        mll_fct, model, solver, bounds, constraints = g_util.init_gp_model(opt_dimension, device, x_data, y_data)

        # BO loop
        print("\n== Start optimization process ==")
        for iteration in range(nb_iter_bo):
            # Fit GP model
            botorch.fit_gpytorch_model(mll=mll_fct)

            # Define the acquisition function
            acq_fct = ExpectedImprovement(model=model, best_f=best_f[-1], maximize=False)
            acq_fct.to(device)

            # Get new candidate
            new_x = joint_optimize_manifold(acq_fct, opt_manifold, solver, q=1, num_restarts=5, raw_samples=100,
                                            bounds=bounds,
                                            pre_processing_manifold=None,
                                            post_processing_manifold=None,
                                            approx_hessian=False, inequality_constraints=constraints)

            new_x_cat = torch.cat([determined_joint, new_x[0]]) 

            # Get new observation
            err = get_pose_error(target_pose, cur_point=new_x_cat)
            new_y = torch.tensor(err)[None]
            new_y = new_y.to(device)
            
            # Update training points
            x_data = torch.cat((x_data, new_x))
            y_data = torch.cat((y_data, new_y))

            # Update best observation
            new_best_f, index = y_data.min(0)
            best_x.append(x_data[index])
            best_f.append(new_best_f)

            # Update the model
            model.set_train_data(x_data, y_data, strict=False)  # strict False necessary to add datapoints
            print("Iteration " + str(iteration) + "\t Best error " + str(new_best_f.item()))
            print(f"\t>> New error : {new_y.item()}")
            
            if new_best_f.item() < 0.2:
                break
        
        # Convert x, y point to radian angle
        joint_point = torch.cat([determined_joint, best_x[-1]])
        joint_angle = g_util.convert_point_to_angle_torch(joint_point)

        return joint_angle

class Baxter:
    left_e0_fixed_offset = Transform(rot=[0.5, 0.5, 0.5, 0.5], pos=[0.107, 0.,    0.   ])
    left_w0_fixed_offset = Transform(rot=[0.5, 0.5, 0.5, 0.5], pos=[0.088, 0.,    0.   ])
    right_e0_fixed_offset = Transform(rot=[0.5, 0.5, 0.5, 0.5], pos=[0.107, 0.,    0.   ])
    right_w0_fixed_offset = Transform(rot=[0.5, 0.5, 0.5, 0.5], pos=[0.088, 0.,    0.   ])

    @staticmethod
    def add_visual_link(link_transforms, f):
        if "left_lower_shoulder" in f.link.name:
            link_transforms["left_upper_elbow_visual"] = np.dot(link_transforms["left_lower_shoulder"],
                                                                        Baxter.left_e0_fixed_offset)
        if "left_lower_elbow" in f.link.name:
            link_transforms["left_upper_forearm_visual"] = np.dot(link_transforms["left_lower_elbow"],
                                                                        Baxter.left_w0_fixed_offset)
        if "right_lower_shoulder" in f.link.name:
            link_transforms["right_upper_elbow_visual"] = np.dot(link_transforms["right_lower_shoulder"],
                                                                        Baxter.right_e0_fixed_offset)
        if "right_lower_elbow" in f.link.name:
            link_transforms["right_upper_forearm_visual"] = np.dot(link_transforms["right_lower_elbow"], 
                                                                        Baxter.right_w0_fixed_offset)
