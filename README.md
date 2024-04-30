# BiConMP MuJoCo

MuJoCo wrapper for BiConMP solver in a docker container.

---

## Install

This project uses VS Code devcontainer. To run this workspace, please first install VS Code with the following extensions:
- Dev Containers (ms-vscode-remote.remote-containers)
- Remote Developement (ms-vscode-remote.vscode-remote-extensionpack)
- Docker (ms-azuretools.vscode-docker)

After that is done, clone this repository onto your developement PC.

```bash
git clone --recurse-submodules https://github.com/Atarilab/biconmp_mujoco.git
```

Then, enter the directory downloaded and open it in vscode

```bash
cd biconmp_mujoco && code .
```

After VS Code has started, there should be a prompt to start this workspace as a container.
Otherwise, you can also do *ctrl + shift + p* then select Dev Container: Rebuild and Reopen in Container to start it manually.

The environment is set up automatically with [BiConMP](https://github.com/machines-in-motion/biconvex_mpc) installed.


## Wrapper interface

This repo provides a wrapper to interface [MuJoCo](https://mujoco.org/) for the simulation environment and [Pinocchio](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/) used by [BiConMP](https://github.com/machines-in-motion/biconvex_mpc). The source code is in `project/example/mj_pin_wrapper`.

The wrapper is imported as a **submodule** in the project from this [repository](https://github.com/Atarilab/mj_pin_wrapper.git).

#### Pinocchio + MuJoCo interface

Defined in the [`RobotWrapperAbstract`](project/example/mj_pin_wrapper/abstract/robot.py) class. This class provides tools to interface model descriptions from both MuJoCo and Pinocchio. It also checks that both descriptions are similar.

Paths of `urdf` and `mjcf` (as `xml`) description files needs to be provided. One can use [`RobotModelLoader`](project/example/mj_pin_wrapper/sim_env/utils.py) to get the paths automatically given the robot name (now only working with `go2` robot).

- `get_pin_state` and `get_mj_state` return the state of robot respectively in Pinocchio and MuJoCo format (different quaternion representation). One can also access to the joint names or end effector names.

- `get_mj_eeff_contact_with_floor` returns which end effectors is in contact with the floor in the simulation as a map {end effector name : True/False}


[`QuadrupedWrapperAbstract`](project/example/mj_pin_wrapper/abstract/robot.py) inherits from [`RobotWrapperAbstract`](project/example/mj_pin_wrapper/abstract/robot.py) and provide additional tools for quadruped robots by abstracting the robot description.

- `get_pin_feet_position_world` returns the feet positions in world frame (same for `hip`, `thigh` and `calf`).





#### Simulator

For custom usage of the wrapper, two abstract classes used in the [`Simulator`](project/example/mj_pin_wrapper/simulator.py) need to be inherited.


1. [`ControllerAbstract`](project/example/mj_pin_wrapper/abstract/controller.py)
Implements the **robot controller or policy**. The following method should be inherited. 
    - `get_torques(q, v, robot_data)`
    This method is called every simulation step by the simulator.
    It takes as input the robot position state `q: np.array`, the robot velocity state `v: np.array` and the simulation data `robot_data: MjData`.
    It should return  the torques for each actuator as a map {joint name : $\tau$}. One can find the joint names the `RobotWrapperAbstract`.

    See [`BiConMPC`](project/example/mpc_controller/bicon_mpc.py) for exemple of inheritance.

2. [`DataRecorderAbstract`](project/example/mj_pin_wrapper/abstract/controller.py)
Implements a class to **record the data from the simulation**. The following method should be inherited.
    - `record(q, v, robot_data)`
    This method is called every simulation step by the simulator.
    It takes as input the robot position state `q: np.array`, the robot velocity state `v: np.array` and the simulation data `robot_data: MjData`.

One can run the simulation with or without viewer using the `run` method.

#### Usage

See [`main.py`](project/example/main.py) for a minimal example.
