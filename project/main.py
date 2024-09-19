import copy
import numpy as np

from mpc_controller.motions.cyclic.go2_trot import trot
from mpc_controller.motions.cyclic.go2_jump import jump
from mpc_controller.motions.cyclic.go2_bound import bound
from mpc_controller.bicon_mpc import BiConMPC
from mpc_controller.mpc_data_recorder import MPCDataRecorder

from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from mj_pin_wrapper.mj_pin_robot import MJPinQuadRobotWrapper
from mj_pin_wrapper.simulator import Simulator
from utils.visuals import desired_contact_locations_callback
from utils.config import Go2Config

if __name__ == "__main__":
    
    ###### Robot model
    cfg = Go2Config
    robot = MJPinQuadRobotWrapper(
        *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
        rotor_inertia=cfg.rotor_inertia,
        gear_ratio=cfg.gear_ratio,
        )
    robot.pin.info()
    robot.mj.info()

    ###### Controller
    controller = BiConMPC(robot.pin, replanning_time=0.05, sim_opt_lag=False)
    # Set command
    v_des, w_des = np.array([0.3, 0., 0.]), 0
    controller.set_command(v_des, w_des)
    # Set gait
    controller.set_gait_params(trot)  # Choose between trot, jump and bound
    
    ###### Data recorder
    data_recorder = MPCDataRecorder(robot, controller, "./data")

    ###### Simulator
    simulator = Simulator(robot.mj, controller, data_recorder)
    # Visualize contact locations
    visual_callback = (lambda viewer, step, q, v, data :
        desired_contact_locations_callback(viewer, step, q, v, data, controller))
    # Run simulation
    sim_time = 5 #s
    simulator.run(
        simulation_time=sim_time,
        use_viewer=True,
        real_time=False,
        visual_callback_fn=visual_callback,
        force_duration=0.75,
        force_intensity=10.
    )
    
    data_recorder.save()