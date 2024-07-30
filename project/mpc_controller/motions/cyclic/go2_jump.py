## Contains go2 12 gait params
## Author : Avadesh Meduri
## Date : 7/7/21

import numpy as np
from mpc_controller.motions.weight_abstract import BiconvexMotionParams

N_JOINTS = 12

#### jump #########################################
jump = BiconvexMotionParams("go2", "Jump")

#########
######### Gait parameters
#########

# Gait horizon
jump.gait_horizon = 1.

# Gait period (s)
jump.gait_period = 0.5
# Gait stance percent [0,1] [FR, FL, RR, RL]
jump.stance_percent = 4*[0.4]
# Gait dt
jump.gait_dt = 0.05
# Gait offset between legs [0,1] [FR, FL, RR, RL]
jump.phase_offset = [0., 0., 0., 0.]
# Gait step height
jump.step_ht = 0.055
# Gait mean/nominal height
jump.nom_ht = 0.35

# Gains toque controller
jump.kp = 17.
jump.kd = 0.2

# ADMM constraints violation norm
jump.rho = 4e+4

#########
######### Kinematic solver
#########

### State
jump.state_wt =  np.array(
    # position (x, y, z)
    [10., 10., 2e2] +
    # orientation (r, p, y)
    [5e3, 4e3, 1e3] +
    # joint positions                    
    [100., 25., 60.]  * 4 +
    # linear velocities (x, y, z)                 
    [10., 10., 2e2] +
    # angular velocities (x, y, z) 
    [9e3, 4e3, 1e3] +
    # joint velocities          
    [10., 35., 25.]  * 4
    )

### Control
jump.ctrl_wt = np.array(
    # force (x, y, z)
    [1e2, 1e2, 1e3] +
    # moment at base (x, y, z)                    
    [2e3, 2e3, 1e3] +
    # torques                 
    [10.0] * N_JOINTS
    )

### Tracking swing end effectors (same for all end effectors swinging)
jump.swing_wt = np.array(
    # contact (x, y, z)
    [2e5, 2e5, 2e4,] +
    # swing (x, y, z)   
    [6e4, 6e4, 2e5,]
    )

### Centroidal
jump.cent_wt = np.array(
    # center of mass (x, y, z)
    3*[1e+1] +
    # linear momentum of CoM (x, y, z)      
    3*[3e+2,] +
    # angular momentum around CoM (x, y, z)       
    3*[7e+2,]         
    )

### Regularization, scale state_wt and ctrl_wt
jump.reg_wt = [
    2.e-1,
    9.e-6
    ]

#########
######### Dynamics solver
#########

### State:
jump.W_X = np.array(
    # centroidal center of mass (x, y, z)
    [1e2, 1e2, 1e+4] +
    # linear momentum of CoM (x, y, z)                    
    [1e+1, 1e+1, 4e+2] +
    # angular momentum around CoM (x, y, z)                    
    [6e+4, 6e+4, 2e4] 
    )

### Terminal state:
jump.W_X_ter = np.array(
    # centroidal center of mass (x, y, z)
    [1e+4, 1e+4, 1e+5] +
    # linear momentum of CoM (x, y, z)                    
    [1e+2, 1e+2, 2e+3] +
    # angular momentum around CoM (x, y, z)                    
    [1e+6, 1e+6, 1e+5]
    )

### Force on each end effectors
jump.W_F = np.array(4*[1e+0, 1e+0, 2.5e+1])

# Maximum force to apply (will be multiplied by the robot weight)
jump.f_max = np.array([.3, .2, .4])

jump.dyn_bound = np.array(3 * [0.45])

### Orientation correction (weights) modifies angular momentum
jump.ori_correction = [0.4, 1., 0.4]