import numpy as np
from math import tan, sin, cos, sqrt

def move(x, dt, u, wheelbase):
    lon_speed = u[0]
    steering_angle = u[1]

    dist = lon_speed * dt

    hdg = x[2]

    if abs(steering_angle) > 0.0001: # is robot turning?
        beta = (dist / wheelbase) * tan(steering_angle)
        r = wheelbase/tan(steering_angle) # radius

        sinh, sinhb = sin(hdg), sin(hdg + beta)
        cosh, coshb = cos(hdg), cos(hdg + beta)
        ret =  x + np.array([-r*sinh + r*sinhb, r*cosh - r*coshb, beta, 0])
        ret[3] = lon_speed
        return ret
    else: # moving in straight line
        ret = x + np.array([dist*cos(hdg), dist*sin(hdg), 0, 0])
        ret[3] = lon_speed
        return ret

def move_with_acc(x, dt, u, wheelbase, debug=False):
    lon_speed = u[0]
    steering_angle = u[1]
    x_acc_veh = u[2]

    dist = ((1.0/2.0)*x_acc_veh*(dt**2)) + (x[3]*dt)

    hdg = x[2]

    if abs(steering_angle) > 0.02: # is robot turning?
        beta = (dist / wheelbase) * tan(steering_angle)
        r = wheelbase/tan(steering_angle) # radius
        

        sinh, sinhb = sin(hdg), sin(hdg + beta)
        cosh, coshb = cos(hdg), cos(hdg + beta)
        ret =  x + np.array([-r*sinh + r*sinhb, r*cosh - r*coshb, beta, x_acc_veh*dt])
        if ret[3] < 0.0: #Velocity condition
            ret[3] = 0.0
        return ret
    else: # moving in straight line
        ret = x + np.array([dist*cos(hdg), dist*sin(hdg), 0, x_acc_veh*dt])
        if ret[3] < 0.0: #Velocity condition
            ret[3] = 0.0
        return ret

def move_with_acc_new(x, dt, u, wheelbase, debug=False):
    lon_speed = u[0]
    steering_angle = u[1]
    x_acc_veh = u[2]

    dist = ((1.0/2.0)*x_acc_veh*(dt**2)) + (x[3]*dt)
    v = x_acc_veh*dt + x[3]

    hdg = x[2]

    if abs(steering_angle) > 0.0000002: # is robot turning?
        beta = np.arctan((1/2) * np.tan(steering_angle))
        ret =  x + np.array([v*np.cos(hdg+beta)*dt, v*np.sin(hdg+beta)*dt, (v*np.cos(beta)*np.tan(steering_angle)/wheelbase)*dt, x_acc_veh*dt])

        if ret[3] < 0.0: #Velocity condition
            ret[3] = 0.0
        return ret
    else: # moving in straight line
        ret = x + np.array([dist*cos(hdg), dist*sin(hdg), 0, x_acc_veh*dt])
        if ret[3] < 0.0: #Velocity condition
            ret[3] = 0.0
        return ret

def normalize_angle(x):
  x = x % (2 * np.pi)  # force in range [0, 2 pi)
  if x > np.pi:  # move to [-pi, pi)
    x -= 2 * np.pi
  return x

