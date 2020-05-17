import numpy as np
from math import tan, sin, cos, sqrt

def move_with_acc(x, dt, u, wheelbase, debug=False):
    steering_angle = u[0]
    x_acc_veh = u[1]

    dist = (0.5*x_acc_veh*(dt**2)) + (x[3]*dt)

    hdg = x[2]

    if abs(steering_angle) > np.deg2rad(0.5): # is robot turning?
        beta = (dist / wheelbase) * tan(steering_angle)
        r = wheelbase/tan(steering_angle) # radius
        

        sinh, sinhb = sin(hdg), sin(hdg + beta)
        cosh, coshb = cos(hdg), cos(hdg + beta)
        ret =  x + np.array([-r*sinh + r*sinhb, r*cosh - r*coshb, beta, x_acc_veh*dt])
        ret[2] = normalize_angle(ret[2])
        if ret[3] < 0.0: #Velocity condition
            ret[3] = 0.0
        return ret
    else: # moving in straight line
        ret = x + np.array([dist*cos(hdg), dist*sin(hdg), 0, x_acc_veh*dt])
        ret[2] = normalize_angle(ret[2])
        if ret[3] < 0.0: #Velocity condition
            ret[3] = 0.0
        return ret

def normalize_angle(x):
  x = x % (2 * np.pi)  # force in range [0, 2 pi)
  if x > np.pi:  # move to [-pi, pi)
    x -= 2 * np.pi
  return x

