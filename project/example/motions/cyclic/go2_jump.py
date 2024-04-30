## Contains go2 12 gait params
## Author : Avadesh Meduri
## Date : 7/7/21

import numpy as np
from motions.weight_abstract import BiconvexMotionParams

N_JOINTS = 12

#### jump #########################################
jump = BiconvexMotionParams("go2", "Jump")

#########
######### Gait parameters
#########

# Gait horizon
jump.gait_horizon = 1.2

# Gait period (s)
jump.gait_period = 0.5
# Gait stance percent [0,1] [FR, FL, RR, RL]
jump.stance_percent = 4*[0.4]
# Gait dt
jump.gait_dt = 0.05
# Gait offset between legs [0,1] [FR, FL, RR, RL]
jump.phase_offset = [0., 0., 0., 0.]
# Gait step height
jump.step_ht = 0.075
# Gait mean/nominal height
jump.nom_ht = 0.34

# Gains toque controller
jump.kp = 18.5
jump.kd = 0.5

# ADMM constraints violation norm
jump.rho = 4e+4

#########
######### Kinematic solver
#########

### State
jump.state_wt = np.array(
    # position (x, y, z)
    [10., 10., 5e2] +
    # orientation (r, p, y)                      
    [8e2, 2e3, 8e2] +
    # joint positions                    
    [10.] * N_JOINTS +
    # linear velocities (x, y, z)                 
    [10., 10., 10.] +
    # anuglar velocities (x, y, z)                       
    [4e2, 5e2, 4e2] +
    # joint velocities          
    [100.0]  *N_JOINTS                      
    )

### Control
jump.ctrl_wt = (
    # force (x, y, z)
    [100., 100., 2e3] +
    # moment at base (x, y, z)                    
    [2e2, 8e2, 2e2] +
    # torques                 
    [10.0] * N_JOINTS                      
    )

### Tracking swing end effectors (same for all end effectors swinging)
jump.swing_wt = [
    # position (x, y, z)
    3*[1e4,],
    # velocities (x, y, z)                         
    3*[1e4,]                                
    ]

### Centroidal
jump.cent_wt = [
    # center of mass (x, y, z)
    3*[0*5e+2,],
    # linear momentum of CoM (x, y, z)                            
    3*[5e+2,] +
    # angular momentum around CoM (x, y, z)                             
    3*[5e+2,]                               
    ]

### Regularization, scale state_wt and ctrl_wt
jump.reg_wt = [
    7e-2,
    1e-5
    ]

#########
######### Dynamics solver
#########

### State:
jump.W_X = np.array(
    # centroidal center of mass (x, y, z)
    [1e-5, 1e-5, 1e+5] +
    # linear momentum of CoM (x, y, z)                    
    [1e+1, 1e+1, 2e+2] +
    # angular momentum around CoM (x, y, z)                    
    [1e+4, 1e+4, 1e4]                       
    )

### Terminal state:
jump.W_X_ter = np.array(
    # centroidal center of mass (x, y, z)
    [1e+6, 1e-4, 1e+6] +
    # linear momentum of CoM (x, y, z)                    
    [1e+2, 1e+2, 2e+3] +
    # angular momentum around CoM (x, y, z)                    
    [1e+6, 1e+6, 1e+6]                      
    )

### Force on each end effectors
jump.W_F = np.array(4*[1e+1, 1e+1, 1.5e+1])

# Maximum force to apply (will be multiplied by the robot weight)
jump.f_max = np.array([1., 1., 3.])

jump.dyn_bound = np.array(3 * [0.45])

### Orientation correction (weights) modifies angular momentum
jump.ori_correction = [0.4, 1.0, 0.6]


