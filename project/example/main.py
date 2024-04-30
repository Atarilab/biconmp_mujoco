import numpy as np

from motions.cyclic.go2_trot import trot
from motions.cyclic.go2_jump import jump
from motions.cyclic.go2_bound import bound
from mpc_controller.bicon_mpc import BiConMPC

from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from mj_pin_wrapper.abstract.robot import QuadrupedWrapperAbstract
from mj_pin_wrapper.simulator import Simulator

class Go2Config:
    name = "go2"
    mesh_dir = "assets"
    rotor_inertia = 0.5*0.250*(0.09/2)**2
    gear_ratio = 6.33

if __name__ == "__main__":
    
    cfg = Go2Config
    
    ### Robot model
    robot = QuadrupedWrapperAbstract(
        *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
        rotor_inertia=cfg.rotor_inertia,
        gear_ratio=cfg.gear_ratio,
        )

    ### Controller
    controller = BiConMPC(robot, replanning_time=0.05, sim_opt_lag=False)

    v_des = np.array([0., 0., 0.])
    w_des = 0.0

    controller.set_command(v_des, w_des)
    controller.set_gait_params(trot)  # Choose between trot, jump and bound

    ### Simulator
    simulator = Simulator(robot, controller)

    # Run 
    SIM_TIME = 2 #s
    simulator.run(
        simulation_time=SIM_TIME,
        viewer=True,
        real_time=False,
    )