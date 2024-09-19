import os
import pinocchio as pin
import numpy as np

from mj_pin_wrapper.abstract.data_recorder import DataRecorderAbstract
from mj_pin_wrapper.mj_pin_robot import MJPinQuadRobotWrapper
from mpc_controller.bicon_mpc import BiConMPC

### Data recorder
class MPCDataRecorder(DataRecorderAbstract):
    FILE_NAME = "data.npz"
    STATE_Q = "q"
    STATE_V = "v"
    FEET_POS = "feet_pos_w"
    TARGET_CONTACT = "target_cnt_w"
    TIME_TO_CONTACT = "time_to_cnt"
    FEET_CONTACT = "foot_cnt"
    V_DES = "v_des"
    W_DES = "w_des"
    
    def __init__(self,
                 robot: MJPinQuadRobotWrapper,
                 controller: BiConMPC,
                 record_dir: str = "",
                 record_step: int = 2,
                 ) -> None:
        """
        MPCDataRecorder class.

        Args:
            robot (MJQuadRobotWrapper): Pin Robot Wrapper
            record_dir (str, optional): Directory to record data. Defaults to "".
            record_step (int, optional): Record data every <record_step>. Defaults to 2.
        """
        super().__init__(record_dir)
        
        self.mj_robot = robot.mj
        self.controller = controller
        self.pin_robot = controller.robot
        self.record_step = record_step
        
        self._update_record_dir(record_dir)
        
        self.keys = [
            ### Position state
            # [x, y, z, qx, qy, qz, qw, q0, ..., qJ] [19]
            MPCDataRecorder.STATE_Q,
            ### Velocity state
            # [v, w, dq0, ..., dqJ] [18]
            MPCDataRecorder.STATE_V,
            ### Foot position in world
            # [foot0_w, foot1_w, foot2_w, foot3_w] [4, 3]
            MPCDataRecorder.FEET_POS,
            ### Target contact locations in world
            # [target_cnt0_w, ..., target_cnt3_w] [4, 3]
            MPCDataRecorder.TARGET_CONTACT,
            ### Time to reach contact locations
            # [t0, t1, t2, t3] [4]
            MPCDataRecorder.TIME_TO_CONTACT,
            ### Is foot in contact (1: yes)
            # [is_cnt0, ... is_cnt3] [4]
            MPCDataRecorder.FEET_CONTACT,
            ### Desired linear velocity
            # [vx, vy, vz] [3]
            MPCDataRecorder.V_DES,
            ### Desired angular velocity
            # [w_yaw] [1]
            MPCDataRecorder.W_DES,
        ]

        self.reset()
        
    def _get_empty_data_dict(self) -> dict:
        d = {k : [] for k in self.keys}
        return d
        
    def _update_record_dir(self, record_dir:str) -> None:
        os.makedirs(record_dir, exist_ok=True)
        self.saving_file_path = os.path.join(record_dir, MPCDataRecorder.FILE_NAME)

    def reset(self) -> None:
        self.recorded_data = self._get_empty_data_dict()
        self.next_cnt_pos_w = []
        self.next_cnt_abs_time = []
        self.step = 0
            
    @staticmethod
    def transform_3d_points(A_T_B : np.ndarray, points_B : np.ndarray) -> np.ndarray:
        """
        Transforms an array of 3d points expressed in frame B
        to frame A according to the transform from B to A (A_T_B)

        Args:
            A_T_B (_type_): SE(3) transform from frame B to frame A
            p_B (_type_): 3D points expressed in frame B. Shape [N, 3]
        """
        # Add a fourth homogeneous coordinate (1) to each point
        ones = np.ones((points_B.shape[0], 1))
        points_B_homogeneous = np.hstack((points_B, ones))
        # Apply the transformation matrix
        points_A_homogeneous = A_T_B @ points_B_homogeneous.T
        # Convert back to 3D coordinates
        points_A = points_A_homogeneous[:3, :].T
        return points_A
    
    @staticmethod
    def transform_cnt_pos_to_world(q : np.ndarray, cnt_pos_b : np.ndarray) -> np.ndarray:
        """
        Transform contact plan to world frame.
        """
        W_T_B = pin.XYZQUATToSE3(q[:7])
        cnt_pos_b = cnt_pos_b.reshape(-1, 3)
        cnt_pos_w = MPCDataRecorder.transform_3d_points(W_T_B, cnt_pos_b)
        # cnt_pos_b have z expressed in world frame
        cnt_pos_w[:, -1] = cnt_pos_b[:, -1]
        cnt_pos_w = cnt_pos_w.reshape(-1, 4, 3)
        
        return cnt_pos_w
    
    def record(self, q: np.ndarray, v: np.ndarray, **kwargs) -> None:
        """ 
        Record data.
        Called by the simulator.
        """
        if self.step % self.record_step == 0:
                
            mj_data = kwargs.get("mj_data", None)
            t = mj_data.time
            
            # State
            self.recorded_data[MPCDataRecorder.STATE_Q].append(q)
            self.recorded_data[MPCDataRecorder.STATE_V].append(v)
            
            # Desired velocity
            self.recorded_data[MPCDataRecorder.V_DES].append(self.controller.v_des)
            self.recorded_data[MPCDataRecorder.W_DES].append(self.controller.w_des)

            # Feet position world
            feet_pos_w = self.pin_robot.get_foot_pos_world()
            self.recorded_data[MPCDataRecorder.FEET_POS].append(feet_pos_w)
            
            # Foot in contact
            foot_contacts = self.mj_robot.foot_contacts()
            foot_contacts_array = np.array([
                foot_contacts["FL"],
                foot_contacts["FR"],
                foot_contacts["RL"],
                foot_contacts["RR"],
                ])
            self.recorded_data[MPCDataRecorder.FEET_CONTACT].append(foot_contacts_array)
            
            # Target contacts, update contact plan positions when MPC is replanning
            if self.controller.pln_ctr == 0:
                # shape [horizon, 4 (feet), 4 (cnt + pos)] 
                cnt_plan = self.controller.gait_gen.cnt_plan
                is_cnt_plan, cnt_plan_pos_b = np.split(cnt_plan, [1], axis=-1)
                cnt_plan_pos_w = self.transform_cnt_pos_to_world(q, cnt_plan_pos_b)
                
                # Next contact position
                next_cnt_t_index = np.argmax(is_cnt_plan>0, axis=0).reshape(-1)
                self.next_cnt_pos_w = cnt_plan_pos_w[next_cnt_t_index, np.arange(len(next_cnt_t_index)), :]
                
                # Absolute simulation time of the next contact
                gait_dt = self.controller.gait_gen.params.gait_dt
                self.next_cnt_abs_time = next_cnt_t_index * gait_dt + t

            time_to_cnt = np.round(np.clip(self.next_cnt_abs_time - t, 0., np.inf), 3)
            self.recorded_data[MPCDataRecorder.TARGET_CONTACT].append(self.next_cnt_pos_w)
            self.recorded_data[MPCDataRecorder.TIME_TO_CONTACT].append(time_to_cnt)
            
        self.step += 1

    def _append_and_save(self, skip_first, skip_last):
        """ 
        Append new data to existing file and save file.
        """
        N = len(self.recorded_data[MPCDataRecorder.STATE_Q])
        
        # If recording is long enough
        if N - skip_first - skip_last > 0:
            
            # Load data and append if exists
            if os.path.exists(self.saving_file_path):
                # Concatenate data
                data_file = np.load(self.saving_file_path)
                data = {k : data_file[k] for k in self.keys}
                if list(data.keys()) == list(self.recorded_data.keys()):
                    for k, v in self.recorded_data.items():
                        data[k] = np.concatenate(
                            (data[k], v[skip_first:N-skip_last]), axis=0
                        )
            else:
                data = self.recorded_data
            
            # Overrides the file with new data
            np.savez(self.saving_file_path, **data)

    def save(self,
             lock = None,
             skip_first_s : float = 0.,
             skip_last_s : float = 0.,
             ) -> None:
        # Avoid repetitve data and near failure states
        skip_first = int(skip_first_s * self.controller.sim_dt)
        skip_last = int(skip_last_s * self.controller.sim_dt)
        
        if lock:
            with lock:
                self._append_and_save(skip_first, skip_last)
        else:
            self._append_and_save(skip_first, skip_last)

        self.reset()