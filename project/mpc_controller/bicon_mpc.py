import numpy as np
import time
from mujoco._structs import MjData

from mpc_controller.cyclic_gait_gen import CyclicQuadrupedGaitGen
from mpc_controller.robot_id_controller import InverseDynamicsController
from mpc_controller.motions.weight_abstract import BiconvexMotionParams
from mj_pin_wrapper.abstract.robot import RobotWrapperAbstract
from mj_pin_wrapper.abstract.controller import ControllerAbstract

class BiConMPC(ControllerAbstract):
    REPLANNING_TIME = 0.05 # replanning time, s
    SIM_OPT_LAG = False # Take optimization time delay into account

    def __init__(self,
                 robot: RobotWrapperAbstract,
                 **kwargs
                 ) -> None:
        super().__init__(robot, **kwargs)
        
        self.robot = robot
        self.nq = robot.pin_model.nq
        self.nv = robot.pin_model.nv
        
        # Optional arguments
        self.optionals = {
            "replanning_time" : BiConMPC.REPLANNING_TIME,
            "sim_opt_lag" : BiConMPC.SIM_OPT_LAG,
        }
        self.optionals.update(**kwargs)
        for k, v in self.optionals.items(): setattr(self, k, v)
        
        # Gait generator
        self.gait_gen = None
        # Gait parameters
        self.gait_params = None
        # Desired linear velocity (x, y, z)
        self.v_des = None
        # Desired angular velocity (x, y, z)
        self.w_des = None
        # Desired contact plan [H, Neef, 3]
        self.contact_plan_des = []
        
        # Inverse dynamics controller
        self.robot_id_ctrl = InverseDynamicsController(
            robot.pin_robot,
            robot.pin_feet_frame_name)

        # MPC timings parameters
        self.sim_t = 0.0
        self.sim_dt = self.robot.mj_model.opt.timestep
        self.index = 0
        self.step = 0
        self.pln_ctr = 0
        self.horizon = int(self.replanning_time / self.sim_dt)
       
        # Init plans
        self.xs_plan = np.empty((3*self.horizon, self.nq + self.nv), dtype=np.float32)
        self.us_plan = np.empty((3*self.horizon, self.nv), dtype=np.float32)
        self.f_plan = np.empty((3*self.horizon, self.robot.n_eeff*3), dtype=np.float32)
        
        self.set_command()
                
    def set_command(self,
                    v_des: np.array = np.zeros((3,)),
                    w_des: float = 0.,
                    ) -> None:
        """
        Set velocities command in world frame.

        Args:
            v_des (np.array, optional): Linear velocities (x, y, z). Defaults to np.zeros((3,)).
            w_des (float, optional): Angular velocities (x, y, z). Defaults to 0..
        """
        self.v_des = v_des
        self.w_des = w_des
        
    def set_contact_plan(self,
                    contact_plan_des: np.array,
                    ) -> None:
        """
        Set custom contact plan for the defined gait.
        Contact plan is expressed in world frame.
        No custom timings.

        Args:
            contact_plan_des (np.array, optional): Contact plan of shape [H, Neeff, 3].
            with H, the planning horizon, Neeff, the number of end effector.
        """
        self.contact_plan_des = contact_plan_des
    
    def set_gait_params(self,
                        gait_params:BiconvexMotionParams,
                       ) -> None:
        """
        Set gait parameters of the gait generator.

        Args:
            gait_params (BiconvexMotionParams): Custom gait parameters. See BiconvexMotionParams.
        """
        self.gait_params = gait_params
        self.gait_gen = CyclicQuadrupedGaitGen(self.robot, self.gait_params, self.replanning_time)
        self.robot_id_ctrl.set_gains(self.gait_params.kp, self.gait_params.kd)

    def _step(self) -> None:
        self.pln_ctr = int((self.pln_ctr + 1)%(self.replanning_time/self.sim_dt))
        self.index += 1
        self.step += 1
        
    def get_torques(self,
                    q: np.array,
                    v: np.array,
                    robot_data: MjData
                    ) -> dict[float]:
        """
        Return torques from simulation data.

        Args:
            q (np.array): position state (nq)
            v (np.array): velocity state (nv)
            robot_data (MjData): MuJoco simulation robot data

        Returns:
            dict[float]: torque command {joint_name : torque value}
        """
        
        sim_t = round(robot_data.time, 3)

        # Replanning
        if self.pln_ctr == 0:
            pr_st = time.time()
            
            self.xs_plan, self.us_plan, self.f_plan = self.gait_gen.optimize(
                q,
                v,
                sim_t,
                self.v_des,
                self.w_des,
                cnt_plan_des=self.contact_plan_des)

            pr_et = time.time() - pr_st
        
        # Second loop onwards lag is taken into account
        if (
            self.step > 0 and
            self.sim_opt_lag and
            self.step > int(self.replanning_time/self.sim_dt) - 1
            ):
            lag = int((1/self.sim_dt)*(pr_et - pr_st))
            self.index = lag
        # If no lag (self.lag < 0)
        elif (
            not self.sim_opt_lag and
            self.pln_ctr == 0. and
            self.step > int(self.replanning_time/self.sim_dt) - 1
        ):
            self.index = 0

        # Compute torques
        tau = self.robot_id_ctrl.id_joint_torques(
            q,
            v,
            self.xs_plan[self.index][:self.nq].copy(),
            self.xs_plan[self.index][self.nq:].copy(),
            self.us_plan[self.index],
            self.f_plan[self.index],
            [])

        # Create command {joint_name : torque value}
        torque_command = {
            joint_name: torque_value
            for joint_name, torque_value
            in zip(self.robot.pin_joint_names, tau)
        }

        # Increment timing variables
        self._step()
        
        return torque_command