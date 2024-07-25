import numpy as np

from mpc_controller.motions.cyclic.go2_trot import trot
from mpc_controller.motions.cyclic.go2_jump import jump
from mpc_controller.motions.cyclic.go2_bound import bound
from mpc_controller.bicon_mpc import BiConMPC

from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from mj_pin_wrapper.abstract.robot import QuadrupedWrapperAbstract
from mj_pin_wrapper.simulator import Simulator
from utils.visuals import desired_contact_locations_callback

class Go2Config:
    # Name of the robot in robot descriptions repo
    name = "go2"
    # Local mesh dir
    mesh_dir = "assets"
    # Rotor ineretia (optional)
    rotor_inertia = 0.5*0.250*(0.09/2)**2
    # Gear ratio (optional)
    gear_ratio = 6.33


if __name__ == "__main__":
    
    ###### Robot model
    
    cfg = Go2Config
    robot = QuadrupedWrapperAbstract(
        *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
        rotor_inertia=cfg.rotor_inertia,
        gear_ratio=cfg.gear_ratio,
        )

    ###### Controller

    controller = BiConMPC(robot, replanning_time=0.05, sim_opt_lag=False)
    # Set command
    v_des, w_des = np.array([0.1, 0., 0.]), 0
    controller.set_command(v_des, w_des)
    # Set gait
    controller.set_gait_params(trot)  # Choose between trot, jump and bound

    ###### Simulator

    simulator = Simulator(robot, controller)
    # Visualize contact locations
    visual_callback = (lambda viewer, step, q, v, data :
        desired_contact_locations_callback(viewer, step, q, v, data, controller))
    # Run simulation
    SIM_TIME = 10 #s
    simulator.run(
        simulation_time=SIM_TIME,
        viewer=True,
        real_time=False,
        visual_callback_fn=visual_callback,
    )