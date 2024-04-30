## Contains go2 12 gait params
## Author : Avadesh Meduri
## Date : 7/7/21

import numpy as np
from motions.weight_abstract import BiconvexMotionParams

N_JOINTS = 12

#### Trot #########################################
trot = BiconvexMotionParams("go2", "Trot")

#########
######### Gait parameters
#########

# Gait horizon
trot.gait_horizon = 1.2

# Gait period (s)
trot.gait_period = 0.5
# Gait stance percent [0,1] [FR, FL, RR, RL]
trot.stance_percent = [0.6] * 4
# Gait dt
trot.gait_dt = 0.05
# Gait offset between legs [0,1] [FR, FL, RR, RL]
trot.phase_offset = [0., 0.5, 0.5, 0.]
# Gait step height
trot.step_ht = 0.05
# Gait mean/nominal height
trot.nom_ht = 0.3

# Gains toque controller
trot.kp = 20.5
trot.kd = 0.1

# ADMM constraints violation norm
trot.rho = 5e+4

#########
######### Kinematic solver
#########

### State
trot.state_wt = np.array(
    # position (x, y, z)
    [0., 0., 10.] +
    # orientation (r, p, y)                      
    [1e2, 1e3, 1e2] +
    # joint positions                    
    [1.0] * N_JOINTS +
    # linear velocities (x, y, z)                 
    [0., 0., 0.] +
    # anuglar velocities (x, y, z)                       
    [1e2, 1e2, 1e2] +
    # joint velocities          
    [0.5]  * N_JOINTS                      
    )

### Control
trot.ctrl_wt = (
    # force (x, y, z)
    [0., 0., 1e3] +
    # moment at base (x, y, z)                    
    [5e2, 5e2, 5e2] +
    # torques                 
    [1.0] * N_JOINTS                      
    )

### Tracking swing end effectors (same for all end effectors swinging)
trot.swing_wt = [
    # position (x, y, z)
    3*[1e4,],
    # velocities (x, y, z)                         
    3*[1e4,]                                
    ]

### Centroidal
trot.cent_wt = [
    # center of mass (x, y, z)
    3*[0.,],
    # linear momentum of CoM (x, y, z)                            
    3*[5.0e2,] +
    # angular momentum around CoM (x, y, z)                             
    3*[5.0e2,]                               
    ]

### Regularization, scale state_wt and ctrl_wt
trot.reg_wt = [
    5e-2,
    1e-5
    ]

#########
######### Dynamics solver
#########

### State:
trot.W_X = np.array(
    # centroidal center of mass (x, y, z)
    [1e-5, 1e-5, 1e+5] +
    # linear momentum of CoM (x, y, z)                    
    [1e+1, 1e+1, 2e+2] +
    # angular momentum around CoM (x, y, z)                    
    [1e+4, 1e+4, 1e4]                       
    )

### Terminal state:
trot.W_X_ter = np.array(
    # centroidal center of mass (x, y, z)
    [1e-4, 1e-4, 5e+5] +
    # linear momentum of CoM (x, y, z)                    
    [1e+2, 1e+2, 1e+4] +
    # angular momentum around CoM (x, y, z)                    
    [1e+5, 1e+5, 1e+5]                      
    )

### Force on each end effectors
trot.W_F = np.array(4*[1.0e1, 1.0e1, 1.0e1])

# Maximum force to apply (will be multiplied by the robot weight)
trot.f_max = np.array([0.4, 0.4, 1.5])

trot.dyn_bound = np.array(3 * [0.45])

### Orientation correction (weights) modifies angular momentum
trot.ori_correction = [0.3, 0.5, 0.4]