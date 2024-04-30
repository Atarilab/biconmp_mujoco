## This file creates the contact plan for different gaits in an MPC fashion
## Author : Avadesh Meduri & Paarth Shah
## Date : 6/05/2021

import time
import numpy as np
import pinocchio as pin
from inverse_kinematics_cpp import InverseKinematics
from biconvex_mpc_cpp import BiconvexMP, KinoDynMP
from gait_planner_cpp import GaitPlanner
from matplotlib import pyplot as plt

from mj_pin_wrapper.abstract.robot import QuadrupedWrapperAbstract
from motions.weight_abstract import ACyclicMotionParams

class CyclicQuadrupedGaitGen:
    GRAVITY = 9.81

    def __init__(self,
                 robot: QuadrupedWrapperAbstract,
                 gait_params: ACyclicMotionParams,
                 planning_time: float,
                 ) -> None:
        """
        Input:
            robot : robot model (pin and mujoco)
            planning_time : planning frequency
        """
        self.planning_time = planning_time
        self.sim_dt = robot.mj_model.opt.timestep
        # Robot pin model and data
        self.rmodel = robot.pin_model
        self.rdata = robot.pin_data
        # URDF model path
        self.path_urdf = robot.path_urdf
        # Foot size
        self.foot_size = robot.foot_size
        
        # Update pin model
        q0, v0 = robot.get_pin_state()
        self.x_reg = np.concatenate([q0, v0])

        pin.forwardKinematics(self.rmodel, self.rdata, q0, np.zeros(self.rmodel.nv))
        pin.updateFramePlacements(self.rmodel, self.rdata)
        
        pin.crba(self.rmodel, self.rdata, q0)
        self.I_composite_b = self.rdata.Ycrb[1].inertia
        
        # End effectors frames name and id
        self.eeff_names = robot.pin_feet_frame_name
        self.eeff_frame_id = [self.rmodel.getFrameId(eeff_name) for eeff_name in self.eeff_names]
        self.n_eeff = robot.n_eeff

        com_init = np.expand_dims(
            pin.centerOfMass(self.rmodel, self.rdata, q0, np.zeros(self.rmodel.nv)),
            0)

        # Offsets to center of mass in base frame
        R = pin.Quaternion(np.array(q0[3:7])).toRotationMatrix()
        self.offsets = np.round(
            ((robot.get_pin_thigh_position_world() - com_init) @ R),
            3)

        # --- Set up Dynamics ---
        self.mass = pin.computeTotalMass(self.rmodel)

        # For plotting
        self.com_traj = []
        self.xs_traj = []
        self.q_traj = []
        self.v_traj = []
        
        # Init motion params
        self.update_gait_params(gait_params)
        
        self.cnt_plan = np.empty((self.horizon, self.n_eeff, 1 + 3), dtype=np.float32)
        
        # To be compatible with old code
        self.height_map = None
        

    def update_gait_params(self, weight_abstract, ik_hor_ratio = 0.5):
        """
        Updates the gaits
        Input:
            weight_abstract : the parameters of the gaits
            ik_hor_ratio : ik horion/dyn horizon 
        """
        self.params = weight_abstract

        # --- Set up gait parameters ---
        self.gait_planner = GaitPlanner(self.params.gait_period, np.array(self.params.stance_percent), \
                                        np.array(self.params.phase_offset), self.params.step_ht)
        #Different horizon parameterizations; only self.params.gait_horizon works for now
        self.gait_horizon = self.params.gait_horizon
        self.horizon = int(np.round(self.params.gait_horizon*self.params.gait_period/self.params.gait_dt,2))

        # --- Set up Inverse Kinematics ---
        self.ik_horizon = int(np.round(ik_hor_ratio*self.params.gait_horizon*self.params.gait_period/self.params.gait_dt, 2))
        self.dt_arr = np.zeros(self.horizon)

        # kino dyn
        self.kd = KinoDynMP(self.path_urdf, self.mass, self.n_eeff, self.horizon, self.ik_horizon)
        self.kd.set_com_tracking_weight(self.params.cent_wt[0])
        self.kd.set_mom_tracking_weight(self.params.cent_wt[1])
        
        self.ik = self.kd.return_ik()
        self.mp = self.kd.return_dyn()

        self.mp.set_rho(self.params.rho)

        # Set up constraints for Dynamics
        self.bx, self.by, self.bz = self.params.dyn_bound[0], self.params.dyn_bound[1], self.params.dyn_bound[2]
        self.params.f_max *= self.mass * CyclicQuadrupedGaitGen.GRAVITY
        self.fx_max, self.fy_max, self.fz_max = self.params.f_max[0], self.params.f_max[1], self.params.f_max[2]

        # --- Set up other variables ---        
        self.X_nom = np.zeros((9*self.horizon))

        # For interpolation (should be moved to the controller)
        self.size = min(self.ik_horizon, int(self.planning_time/self.params.gait_dt) + 2)
        if self.planning_time > self.params.gait_dt:
            self.size -= 1

            
    def create_cnt_plan(self, q, v, t, v_des, w_des):
        
        com = np.round(pin.centerOfMass(self.rmodel, self.rdata, q, v)[0:2], 3)
        z_height = pin.centerOfMass(self.rmodel, self.rdata, q, v)[2]
        #vcom = np.round(v[0:3], 3)

        vtrack = v_des[0:2] # this effects the step location (if set to vcom it becomes raibert)
        #vtrack = vcom[0:2]
        
        # This array determines when the swing foot cost should be enforced in the ik
        self.swing_time = np.zeros((self.horizon, self.n_eeff))
        self.curr_cnt = np.zeros(self.n_eeff)
        # Contact Plan Matrix: horizon x num_eef x 4: The '4' gives the contact plan and location:
        # i.e. the last vector should be [1/0, x, y, z] where 1/0 gives a boolean for contact (1 = contact, 0 = no cnt)

    
        # Get current orientation
        R = pin.Quaternion(np.array(q[3:7])).toRotationMatrix()
        
        # Express in base frame
        v_des = R @ v_des
        
        # Set roll and pitch to 0.
        rpy_vector = pin.rpy.matrixToRpy(R)
        rpy_vector[0] = 0.0
        rpy_vector[1] = 0.0
        R = pin.rpy.rpyToMatrix(rpy_vector)
        
        for i in range(self.horizon):
            for j in range(self.n_eeff):
                # First time step
                if i == 0:
                    # Contact
                    if self.gait_planner.get_phase(t, j) == 1:
                        self.cnt_plan[i][j][0] = 1
                        self.cnt_plan[i][j][1:4] = np.round(self.rdata.oMf[self.eeff_frame_id[j]].translation, 3)

                    # No contact
                    else:
                        self.cnt_plan[i][j][0] = 0
                        self.cnt_plan[i][j][1:4] = np.round(self.rdata.oMf[self.eeff_frame_id[j]].translation, 3)
                
                # Other time steps
                else:
                    ft = np.round(t + i*self.params.gait_dt, 3)

                    # Contact
                    if self.gait_planner.get_phase(ft, j) == 1:
                        self.cnt_plan[i][j][0] = 1
                        
                        # If stays in contact, same contact location
                        if self.cnt_plan[i-1][j][0] == 1:
                            self.cnt_plan[i][j][1:4] = self.cnt_plan[i-1][j][1:4]
                        
                        else:
                            hip_loc = com + np.matmul(R, self.offsets[j])[0:2] + i*self.params.gait_dt*vtrack
                            raibert_step = 0.5*vtrack*self.params.gait_period*self.params.stance_percent[j] - 0.05*(vtrack - v_des[0:2])
                            ang_step = 0.5*np.sqrt(z_height/CyclicQuadrupedGaitGen.GRAVITY)*vtrack
                            ang_step = np.cross(ang_step, [0.0, 0.0, w_des])
                        
                            self.cnt_plan[i][j][1:3] = raibert_step[0:2] + hip_loc + ang_step[0:2]

                            if self.height_map != None:
                                self.cnt_plan[i][j][3] = self.height_map.getHeight(self.cnt_plan[i][j][1], self.cnt_plan[i][j][2])
                            
                            self.cnt_plan[i][j][3] = self.foot_size
                        
                    else:
                        #If foot will not be in contact
                        self.cnt_plan[i][j][0] = 0
                        per_ph = np.round(self.gait_planner.get_percent_in_phase(ft, j), 3)
                        hip_loc = com + np.matmul(R,self.offsets[j])[0:2] + i*self.params.gait_dt*vtrack
                        ang_step = 0.5*np.sqrt(z_height/CyclicQuadrupedGaitGen.GRAVITY)*vtrack
                        ang_step = np.cross(ang_step, [0.0, 0.0, w_des])
                        
                        if per_ph < 0.5:
                            self.cnt_plan[i][j][1:3] = hip_loc + ang_step[0:2]
                        else:
                            raibert_step = 0.5*vtrack*self.params.gait_period*self.params.stance_percent[j] - 0.05*(vtrack - v_des[0:2])
                            self.cnt_plan[i][j][1:3] = hip_loc + ang_step[0:2]

                        #What is this?
                        if per_ph - 0.5 < 0.02:
                            self.swing_time[i][j] = 1

                        if self.height_map != None:
                            self.cnt_plan[i][j][3] = self.height_map.getHeight(self.cnt_plan[i][j][1], self.cnt_plan[i][j][2])
                        
                        self.cnt_plan[i][j][3] = self.foot_size
                    
            if i == 0:
                dt = self.params.gait_dt - np.round(np.remainder(t,self.params.gait_dt),2)
                if dt == 0:
                    dt = self.params.gait_dt
            else:
                dt = self.params.gait_dt
            self.mp.set_contact_plan(self.cnt_plan[i], dt)
            self.dt_arr[i] = dt

        return self.cnt_plan
    
    def follow_contact_plan(self, q:np.array, time:float, cnt_plan_des_world:list):
        """
        Update the contact plan given some desired foot locations
        in world frame.
        

        Args:
            q (array): current position state of the robot
            time (float): time (s)
            cnt_plan_des_world (list): desired contact location
        """
        # This array determines when the swing foot cost should be enforced in the ik
        self.swing_time = np.zeros((self.horizon, self.n_eeff))
        self.curr_cnt = np.zeros(self.n_eeff)
        cnt_plan_des_world = np.array(cnt_plan_des_world)

        # Contact Plan Matrix: horizon x num_eef x 4: The '4' gives the contact plan and location:
        # i.e. the last vector should be [1/0, x, y, z] where 1/0 gives a boolean for contact (1 = contact, 0 = no cnt)

        # R base in world frame
        w_R_b = pin.Quaternion(np.array(q[3:7])).toRotationMatrix()
        # Set roll and pitch to 0.
        rpy_vector = pin.rpy.matrixToRpy(w_R_b)
        rpy_vector[0] = 0.0
        rpy_vector[1] = 0.0
        # R world in base frame
        b_R_w = pin.rpy.rpyToMatrix(rpy_vector).T

        # If custom contact plan is given
        if len(cnt_plan_des_world) >= self.horizon:
            for i in range(self.horizon):
                for j in range(self.n_eeff):
                    b_pos_contact = b_R_w @ cnt_plan_des_world[i][j]
                    b_pos_contact[-1] = cnt_plan_des_world[i][j][-1] + self.foot_size
             
                    # First time step
                    if i == 0:
                        self.cnt_plan[i][j][1:4] = b_pos_contact
                        # Contact
                        if self.gait_planner.get_phase(time, j) == 1:
                            self.cnt_plan[i][j][0] = 1
                        else:
                            self.cnt_plan[i][j][0] = 0
                    
                    # Next time steps
                    else:
                        ft = np.round(time + i*self.params.gait_dt, 3)

                        # Contact
                        if self.gait_planner.get_phase(ft, j) == 1:
                            self.cnt_plan[i][j][0] = 1
                            
                            # If stays in contact, same contact location
                            if self.cnt_plan[i-1][j][0] == 1:
                                self.cnt_plan[i][j][1:4] = self.cnt_plan[i-1][j][1:4]
                            
                            else:
                                self.cnt_plan[i][j][1:4] = b_pos_contact
                            
                        # No contact
                        else:
                            self.cnt_plan[i][j][0] = 0                
                            #What is this?
                            per_ph = np.round(self.gait_planner.get_percent_in_phase(ft, j), 3)
                            if np.abs(per_ph - 0.5) < 0.02:
                                self.swing_time[i][j] = 1
                                
                            self.cnt_plan[i][j][1:4] = b_pos_contact
                            
                if i == 0:
                    dt = self.params.gait_dt - np.round(np.remainder(time,self.params.gait_dt),2)
                    if dt == 0:
                        dt = self.params.gait_dt
                else:
                    dt = self.params.gait_dt
                self.mp.set_contact_plan(self.cnt_plan[i], dt)
                self.dt_arr[i] = dt

    def create_costs(self, q, v, v_des, w_des, ori_des):
        """
        Input:
            q : joint positions at current time
            v : joint velocity at current time
            v_des : desired velocity of center of mass
            t : time within the step
        """

        self.x = np.hstack((q,v))

        # --- Set Up IK --- #
        #Right now this is only setup to go for the *next* gait period only
        for i in range(self.ik_horizon):
            for j in range(self.n_eeff):
                if self.cnt_plan[i][j][0] == 1:
                    self.ik.add_position_tracking_task_single(self.eeff_frame_id[j], self.cnt_plan[i][j][1:4], self.params.swing_wt[0],
                                                              "cnt_" + str(0) + self.eeff_names[j], i)
                elif self.swing_time[i][j] == 1:
                    pos = self.cnt_plan[i][j][1:4].copy()
                    pos[2] = self.params.step_ht
                    self.ik.add_position_tracking_task_single(self.eeff_frame_id[j], pos, self.params.swing_wt[1],
                                                              "via_" + str(0) + self.eeff_names[j], i)

        self.ik.add_state_regularization_cost(0, self.ik_horizon, self.params.reg_wt[0], "xReg", self.params.state_wt, self.x_reg, False)
        self.ik.add_ctrl_regularization_cost(0, self.ik_horizon, self.params.reg_wt[1], "uReg", self.params.ctrl_wt, np.zeros(self.rmodel.nv), False)

        self.ik.add_state_regularization_cost(0, self.ik_horizon, self.params.reg_wt[0], "xReg", self.params.state_wt, self.x_reg, True)
        self.ik.add_ctrl_regularization_cost(0, self.ik_horizon, self.params.reg_wt[1], "uReg", self.params.ctrl_wt, np.zeros(self.rmodel.nv), True)

        self.ik.setup_costs(self.dt_arr[0:self.ik_horizon])

        # --- Setup Dynamics --- #

        # initial and terminal state
        self.X_init = np.zeros(9)
        X_ter = np.zeros_like(self.X_init)
        pin.computeCentroidalMomentum(self.rmodel, self.rdata)
        self.X_init[0:3] = pin.centerOfMass(self.rmodel, self.rdata, q.copy(), v.copy())
        self.X_init[3:] = np.array(self.rdata.hg)
        self.X_init[3:6] /= self.mass

        self.X_nom[0::9] = self.X_init[0]
        for i in range(1, self.horizon):
            self.X_nom[9*i+0] = self.X_nom[9*(i-1)+0] + v_des[0]*self.dt_arr[i]
            self.X_nom[9*i+1] = self.X_nom[9*(i-1)+1] + v_des[1]*self.dt_arr[i]

        self.X_nom[2::9] = self.params.nom_ht
        self.X_nom[3::9] = v_des[0]
        self.X_nom[4::9] = v_des[1]
        self.X_nom[5::9] = v_des[2]

        #Compute angular momentum / orientation correction
        R = pin.Quaternion(np.array(ori_des)).toRotationMatrix()
        rpy_vector = pin.rpy.matrixToRpy(R)
        rpy_vector[0] = 0.0
        rpy_vector[1] = 0.0
        des_quat = pin.Quaternion(pin.rpy.rpyToMatrix(rpy_vector))

        amom = self.compute_ori_correction(q, des_quat.coeffs())

        #Set terminal references
        X_ter[0:2] = self.X_init[0:2] + (self.params.gait_horizon*self.params.gait_period*v_des)[0:2] #Changed this
        X_ter[2] = self.params.nom_ht
        X_ter[3:6] = v_des
        X_ter[6:] = amom
        #print("X_terminal: ", X_ter)
        self.X_nom[6::9] = amom[0]*self.params.ori_correction[0]
        self.X_nom[7::9] = amom[1]*self.params.ori_correction[1]

        if w_des == 0:
            self.X_nom[8::9] = amom[2]*self.params.ori_correction[2]
        else:
            yaw_momentum = np.matmul(self.I_composite_b,[0.0, 0.0, w_des])[2]
            self.X_nom[8::9] = yaw_momentum
            X_ter[8] = yaw_momentum
            #print(yaw_momentum)

        # Setup dynamic optimization costs
        bounds = np.tile([-self.bx, -self.by, 0, self.bx, self.by, self.bz], (self.horizon,1))
        #print("Bounds: ", bounds)
        #print("X_nom: ", np.resize(np.asarray(self.X_nom), (10,9)))
        self.mp.create_bound_constraints(bounds, self.fx_max, self.fy_max, self.fz_max)
        self.mp.create_cost_X(np.tile(self.params.W_X, self.horizon), self.params.W_X_ter, X_ter, self.X_nom)
        self.mp.create_cost_F(np.tile(self.params.W_F, self.horizon))

    def compute_ori_correction(self, q, des_quat):
        """
        This function computes the AMOM required to correct for orientation
        q : current joint configuration
        des_quat : desired orientation
        """
        pin_quat = pin.Quaternion(np.array(q[3:7]))
        pin_des_quat = pin.Quaternion(np.array(des_quat))

        omega = pin.log3((pin_des_quat*(pin_quat.inverse())).toRotationMatrix())

        return omega
    
    def _update_pin_data(self, q, v):
        pin.forwardKinematics(self.rmodel, self.rdata, q, v)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        
    def optimize(self, q, v, t, v_des, w_des, cnt_plan_des=[], X_wm = None, F_wm = None, P_wm = None):

        q_origin = np.copy(q)
        q_origin[:2] = 0.
        
        if w_des != 0:
            ori_des = q[3:7]
        else:
            ori_des = np.array([0., 0., 0., 1.])

        self._update_pin_data(q_origin, v)
        
        if len(cnt_plan_des) == 0:
            self.create_cnt_plan(q_origin, v, t, v_des, w_des)
        else:
            self.follow_contact_plan(q, t, cnt_plan_des)
            
        # Creates costs for IK and Dynamics
        self.create_costs(q_origin, v, v_des, w_des, ori_des)

        # pinocchio complains otherwise 
        q = pin.normalize(self.rmodel, q_origin)
        self.kd.optimize(q_origin, v, 100, 1)

        # Results
        #com_opt = self.mp.return_opt_com()
        #mom_opt = self.mp.return_opt_mom()
        F_opt = self.mp.return_opt_f()
        xs = self.ik.get_xs()
        us = self.ik.get_us()

        n_eff_3d = 3*self.n_eeff
        step_dt = int(self.dt_arr[0]/self.sim_dt)
        if not hasattr(self, "f_int"):
            self.f_int = np.empty((self.size * step_dt, n_eff_3d), dtype=np.float32)
            self.xs_int = np.empty((self.size * step_dt, self.rmodel.nv + self.rmodel.nq), dtype=np.float32)
            self.us_int = np.empty((self.size * step_dt, len(us[0])), dtype=np.float32)
            #self.com_int = np.empty((self.size * step_dt, len(com_opt[0, :])), dtype=np.float32)
            #self.mom_int = np.empty((self.size * step_dt, len(mom_opt[0, :])), dtype=np.float32)
        
        for i in range(self.size):
            self.f_int[i * step_dt : (i+1) * step_dt, :] =  np.linspace(F_opt[i*n_eff_3d:n_eff_3d*(i+1)], F_opt[n_eff_3d*(i+1):n_eff_3d*(i+2)], step_dt, dtype=np.float32)
            self.xs_int[i * step_dt : (i+1) * step_dt, :] =  np.linspace(xs[i], xs[i+1], step_dt, dtype=np.float32)
            self.us_int[i * step_dt : (i+1) * step_dt, :] =  np.linspace(us[i], us[i+1], step_dt, dtype=np.float32)
            #self.com_int[i * step_dt : (i+1) * step_dt, :] =  np.linspace(com_opt[i], com_opt[i+1], step_dt, dtype=np.float32)
            #self.mom_int[i * step_dt : (i+1) * step_dt, :] =  np.linspace(mom_opt[i], mom_opt[i+1], step_dt, dtype=np.float32)

        self.q_traj.append(q)
        self.v_traj.append(v)
        self.xs_traj.append(xs)

        return self.xs_int, self.us_int, self.f_int

    def plot(self, q, v, plot_force = True):
        com_opt = self.mp.return_opt_com()
        mom_opt = self.mp.return_opt_mom()
        optimized_forces = self.mp.return_opt_f()
        ik_com_opt = self.ik.return_opt_com()
        ik_mom_opt = self.ik.return_opt_mom()
        com = pin.centerOfMass(self.rmodel, self.rdata, q.copy(), v.copy())

        # Plot Center of Mass
        fig, ax = plt.subplots(3,1)
        ax[0].plot(com_opt[:, 0], label="Dyn com x")
        ax[0].plot(ik_com_opt[:, 0], label="IK com x")
        ax[0].plot(com[0], 'o', label="Current Center of Mass x")
        ax[1].plot(com_opt[:, 1], label="Dyn com y")
        ax[1].plot(ik_com_opt[:, 1], label="IK com y")
        ax[1].plot(com[1], 'o', label="Current Center of Mass y")
        ax[2].plot(com_opt[:, 2], label="Dyn com z")
        ax[2].plot(ik_com_opt[:, 2], label="IK com z")
        ax[2].plot(com[2], 'o', label="Current Center of Mass z")

        ax[0].grid()
        ax[0].legend()
        ax[1].grid()
        ax[1].legend()
        ax[2].grid()
        ax[2].legend()

        # Plot End-Effector Forces
        if plot_force:
            fig, ax_f = plt.subplots(self.n_eeff, 1)
            for n in range(self.n_eeff):
                ax_f[n].plot(optimized_forces[3*n::3*self.n_eeff], label = self.eeff_names[n] + " Fx")
                ax_f[n].plot(optimized_forces[3*n+1::3*self.n_eeff], label = self.eeff_names[n] + " Fy")
                ax_f[n].plot(optimized_forces[3*n+2::3*self.n_eeff], label = self.eeff_names[n] + " Fz")
                ax_f[n].grid()
                ax_f[n].legend()

        # Plot Linear Momentum
        fig, ax_m = plt.subplots(3,1)
        ax_m[0].plot(mom_opt[:, 0], label = "Dyn linear_momentum x")
        ax_m[0].plot(ik_mom_opt[:, 0], label="IK linear_momentum x")
        ax_m[1].plot(mom_opt[:, 1], label = "linear_momentum y")
        ax_m[1].plot(ik_mom_opt[:, 1], label="IK linear_momentum y")
        ax_m[2].plot(mom_opt[:, 2], label = "linear_momentum z")
        ax_m[2].plot(ik_mom_opt[:, 2], label="IK linear_momentum z")
        ax_m[0].grid()
        ax_m[0].legend()

        ax_m[1].grid()
        ax_m[1].legend()

        ax_m[2].grid()
        ax_m[2].legend()

        # Plot Linear Momentum
        fig, ax_am = plt.subplots(3,1)
        ax_am[0].plot(mom_opt[:, 3], label = "Dynamics Angular Momentum around X")
        ax_am[0].plot(ik_mom_opt[:, 3], label="Kinematic Angular Momentum around X")
        ax_am[1].plot(mom_opt[:, 4], label = "Dynamics Angular Momentum around Y")
        ax_am[1].plot(ik_mom_opt[:, 4], label="Kinematic Angular Momentum around Y")
        ax_am[2].plot(mom_opt[:, 5], label = "Dynamics Angular Momentum around Z")
        ax_am[2].plot(ik_mom_opt[:, 5], label="Kinematic Angular Momentum around Z")
        ax_am[0].grid()
        ax_am[0].legend()

        ax_am[1].grid()
        ax_am[1].legend()

        ax_am[2].grid()
        ax_am[2].legend()

        plt.show()

    def plot_joints(self):
        self.xs_traj = np.array(self.xs_traj)
        self.xs_traj = self.xs_traj[:,:,:self.rmodel.nq]
        self.q_traj = np.array(self.q_traj)
        x = self.dt*np.arange(0, len(self.xs_traj[1]) + len(self.xs_traj), 1)
        # com plots
        fig, ax = plt.subplots(3,1)
        for i in range(len(self.xs_traj)):
            st_hor = i*int(self.planning_time/self.dt)
            ax[0].plot(x[st_hor], self.q_traj[i][10], 'o')
            ax[0].plot(x[st_hor:st_hor + len(self.xs_traj[i])], self.xs_traj[i][:,10])

        plt.show()

    def save_plan(self, file_name):
        """
        This function saves the plan for later plotting
        Input:
            file_name : name of the file
        """

        np.savez("./"+file_name, com_opt = self.mp.return_opt_com(),\
                                 mom_opt = self.mp.return_opt_mom(),\
                                 F_opt = self.mp.return_opt_f(), \
                                 ik_com_opt = self.ik.return_opt_com(),\
                                 ik_mom_opt = self.ik.return_opt_mom(),\
                                 xs = self.ik.get_xs(),
                                 us = self.ik.get_us(),
                                 cnt_plan = self.cnt_plan)
                                 
        print("finished saving ...")
        assert False

    def plot_plan(self, q, v, plot_force = True):
        com_opt = self.mp.return_opt_com()
        mom_opt = self.mp.return_opt_mom()
        F_opt = self.mp.return_opt_f()
        ik_com_opt = self.ik.return_opt_com()
        ik_mom_opt = self.ik.return_opt_mom()
        com = pin.centerOfMass(self.rmodel, self.rdata, q.copy(), v.copy())

        # Plot Center of Mass
        fig, ax = plt.subplots(3,1)
        ax[0].plot(com_opt[:, 0], label="Dyn com x")
        ax[0].plot(ik_com_opt[:, 0], label="IK com x")
        ax[0].plot(com[0], 'o', label="Current Center of Mass x")
        ax[1].plot(com_opt[:, 1], label="Dyn com y")
        ax[1].plot(ik_com_opt[:, 1], label="IK com y")
        ax[1].plot(com[1], 'o', label="Current Center of Mass y")
        ax[2].plot(com_opt[:, 2], label="Dyn com z")
        ax[2].plot(ik_com_opt[:, 2], label="IK com z")
        ax[2].plot(com[2], 'o', label="Current Center of Mass z")

        ax[0].grid()
        ax[0].legend()
        ax[1].grid()
        ax[1].legend()
        ax[2].grid()
        ax[2].legend()

        # Plot End-Effector Forces
        if plot_force:
            fig, ax_f = plt.subplots(self.n_eeff, 1)
            for n in range(self.n_eeff):
                ax_f[n].plot(F_opt[3*n::3*self.n_eeff], label = self.eeff_names[n] + " Fx")
                ax_f[n].plot(F_opt[3*n+1::3*self.n_eeff], label = self.eeff_names[n] + " Fy")
                ax_f[n].plot(F_opt[3*n+2::3*self.n_eeff], label = self.eeff_names[n] + " Fz")
                ax_f[n].grid()
                ax_f[n].legend()

        # Plot Momentum
        fig, ax_m = plt.subplots(6,1)
        ax_m[0].plot(mom_opt[:, 0], label = "Dyn linear_momentum x")
        ax_m[0].plot(ik_mom_opt[:, 0], label="IK linear_momentum x")
        ax_m[1].plot(mom_opt[:, 1], label = "linear_momentum y")
        ax_m[1].plot(ik_mom_opt[:, 1], label="Dyn IK linear_momentum y")
        ax_m[2].plot(mom_opt[:, 2], label = "linear_momentum z")
        ax_m[2].plot(ik_mom_opt[:, 2], label="Dyn IK linear_momentum z")
        ax_m[3].plot(mom_opt[:, 3], label = "Dyn Angular momentum x")
        ax_m[3].plot(ik_mom_opt[:, 3], label="IK Angular momentum x")
        ax_m[4].plot(mom_opt[:, 4], label = "Dyn Angular momentum y")
        ax_m[4].plot(ik_mom_opt[:, 4], label="IK Angular momentum y")
        ax_m[5].plot(mom_opt[:, 5], label = "Dyn Angular momentum z")
        ax_m[5].plot(ik_mom_opt[:, 5], label="IK Angular momentum z")
        ax_m[0].grid()
        ax_m[0].legend()
        ax_m[1].grid()
        ax_m[1].legend()
        ax_m[2].grid()
        ax_m[2].legend()
        ax_m[3].grid()
        ax_m[3].legend()
        ax_m[4].grid()
        ax_m[4].legend()
        ax_m[5].grid()
        ax_m[5].legend()

        plt.show()
