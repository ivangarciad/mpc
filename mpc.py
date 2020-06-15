from transforms3d.euler import euler2mat, mat2euler
from scipy import optimize
import numpy as np
from warnings import filterwarnings
from sympy import *
import matplotlib.pyplot as plt
from sympy.geometry import *
import model_carla as model
import utils


class MPC:
    def __init__(self):
       print ('MPC process' )

    def mpc_process(self, x, *args):
      x_state_list = [args[0]]

      dt = args[1]
      lf = args[2]
      N = args[3]

      poly = args[4]
      coefficients = args[5]
      label = args[6]
      v_target = args[7]*np.ones(N) # m/s

      for i in range(N):
        x_state_list = np.append(x_state_list, [model.move_with_acc(np.asarray(x_state_list[-1]), dt[i], np.asarray([x[i + N], x[i]]), lf)], axis=0)

      #print ('Mpc Predicted State')
      #print (x_state_list)

      method_1 = True

      if label == 'y_poly':
        x_target = x_state_list[1:,0]
        y_target = poly(x_target)
        coefficients_der = np.polyder(coefficients)
        t_target = np.poly1d(coefficients_der)(x_target)
      
      
        if method_1 == True:
          d_lateral = [[(x_state_list[1,1] - y_target[0])/np.cos(np.pi - x_state_list[1,2])]]
          #print ('y_poly: ' + str(d_lateral))
          
          for y_target_elem, yt_elem, yawt_elem in zip(y_target[1:], x_state_list[2:,1], x_state_list[2:,2]):
              d_lateral = np.append(d_lateral, [[(yt_elem - y_target_elem)/np.cos(np.pi-yawt_elem)]])
      
      elif label == 'x_poly':
        y_target = x_state_list[1:,1]
        x_target = poly(y_target)
        coefficients_der = np.polyder(coefficients)
        t_target = np.poly1d(coefficients_der)(y_target)
      
        if method_1 == True:
          d_lateral = [[(x_state_list[1,0] - x_target[0])/np.sin(np.pi - x_state_list[1,2])]]
          #print ('x_poly: ' + str(d_lateral))
          
          for x_target_elem, xt_elem, yawt_elem in zip(x_target[1:], x_state_list[2:,0], x_state_list[2:,2]):
              d_lateral = np.append(d_lateral, [[(xt_elem - x_target_elem)/np.sin(np.pi - yawt_elem)]])

      if method_1 == False:
        for i in range(1,N+1):
          car_vector = {'x': x_state_list[i,0], 'y': x_state_list[i,1], 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': x_state_list[i,2]}
          car_matrix = utils.vector_to_matrix_pose(car_vector)
          
          ref_vector = {'x': x_target[i-1], 'y': y_target[i-1], 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': t_target[i-1]}
          ref_matrix = utils.vector_to_matrix_pose(ref_vector)
          
          if label == 'y_poly':
            diff_matrix = np.dot(np.linalg.inv(ref_matrix), car_matrix)
          else:
            diff_matrix = np.dot(np.linalg.inv(car_matrix), ref_matrix)
        
          diff_vector = utils.matrix_to_vector_pose(diff_matrix)
        
          if i == 1:
            d_lateral_new = [[diff_vector['y']]]
          else:
            d_lateral_new = np.append(d_lateral_new, [[diff_vector['y']]])

      et = [[x_state_list[1,2] - t_target[0]]]
      for t_target_elem, tt_elem in zip(t_target[1:], x_state_list[2:,2]):
          et = np.append(et, [[tt_elem - t_target_elem]], axis=0)

      ev = [[x_state_list[1,3] - v_target[0]]]
      for v_target_elem, vt_elem in zip(v_target[1:], x_state_list[2:,3]):
          ev = np.append(ev, [[vt_elem - v_target_elem]], axis=0)

      error = 0
      for d_lateral_elem in zip(d_lateral):
          error += 900*np.linalg.norm(d_lateral_elem[0])

      for et_elem in zip(et):
          error += 320*np.linalg.norm(et_elem[0])

      for ev_elem in zip(ev):
          error += 300*np.linalg.norm(ev_elem[0])

      for x_elem in x:
          error += 0*np.linalg.norm(x_elem)

      for x_elem_post, x_elem_ant in zip(x[1:N-1], x[:N-2]):  #Acc
          error += 200*np.linalg.norm(x_elem_ant - x_elem_post)

      for x_elem_post, x_elem_ant in zip(x[N+1:], x[N:]): #Steer
          error += 4550*np.linalg.norm(x_elem_ant - x_elem_post) # Resultados muy buenos

      return error 
            
    def opt(self, x_state, dt, wheelbase, acc, steer, N, poly, coefficients, label, v_target):

      print ('Veh speed: ' + str(x_state[3]))
      acc_contraint_max = 2   #m/s^2 
      acc_contraint_min = 3   #m/s^2
      steer_contraint = np.deg2rad(20)
      

      if N == 6:
        cons = ({'type': 'ineq', 'fun': lambda x:  x[0] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[0] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[1] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[1] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[2] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[2] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[3] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[3] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[4] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[4] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[5] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[5] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[6] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[6] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[7] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[7] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[8] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[8] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[9] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[9] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[10] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[10] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[11] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[11] + steer_contraint})
      elif N == 8:
        cons = ({'type': 'ineq', 'fun': lambda x:  x[0] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[0] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[1] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[1] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[2] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[2] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[3] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[3] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[4] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[4] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[5] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[5] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[6] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[6] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[7] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[7] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[8] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[8] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[9] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[9] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[10] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[10] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[11] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[11] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[12] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[12] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[13] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[13] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[14] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[14] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[15] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[15] + steer_contraint})
      elif N == 10:
        cons = ({'type': 'ineq', 'fun': lambda x:  x[0] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[0] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[1] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[1] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[2] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[2] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[3] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[3] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[4] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[4] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[5] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[5] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[6] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[6] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[7] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[7] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[8] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[8] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[9] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[9] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[10] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[10] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[11] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[11] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[12] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[12] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[13] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[13] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[14] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[14] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[15] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[15] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[16] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[16] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[17] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[17] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[18] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[18] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[19] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[19] + steer_contraint})
      elif N == 20:
        cons = ({'type': 'ineq', 'fun': lambda x:  x[0] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[0] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[1] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[1] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[2] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[2] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[3] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[3] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[4] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[4] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[5] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[5] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[6] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[6] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[7] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[7] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[8] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[8] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[9] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[9] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[10] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[10] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[11] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[11] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[12] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[12] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[13] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[13] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[14] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[14] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[15] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[15] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[16] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[16] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[17] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[17] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[18] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[18] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[19] + acc_contraint_min},
                {'type': 'ineq', 'fun': lambda x: -x[19] + acc_contraint_max},
                {'type': 'ineq', 'fun': lambda x:  x[20] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[20] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[21] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[21] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[22] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[22] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[23] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[23] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[24] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[24] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[25] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[25] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[26] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[26] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[27] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[27] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[28] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[28] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[29] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[29] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[30] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[30] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[31] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[31] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[32] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[32] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[33] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[33] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[34] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[34] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[35] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[35] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[36] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[36] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[37] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[37] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[38] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[38] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x:  x[39] + steer_contraint},
                {'type': 'ineq', 'fun': lambda x: -x[39] + steer_contraint})

      x0 = acc*np.ones(N)
      x0 = np.append(x0, steer*np.ones(N))

      #options = {'maxiter': 10000, 'disp': True}
      options = {'maxiter': 150, 'disp': False}

      #sol = optimize.minimize(self.mpc_process, args=(v_pred, t_pred, y_pred, x_pred, dt, wheelbase, N, poly, coefficients), x0=x0, method='trust-constr', bounds=bounds, options=options)
      sol = optimize.minimize(self.mpc_process, args=(x_state, dt, wheelbase, N, poly, coefficients, label, v_target), x0=x0, method='COBYLA', options=options, constraints=cons)
      #print sol


      return sol.x

if __name__== "__main__":
    size = 10
    dt = 0.1*np.ones(size)
    lf = 2

    t_target = np.deg2rad(45)*np.ones(size)
    v_target = 10*np.ones(size)

    x_target = 100*np.ones(size)
    y_target = 100*np.ones(size)

    y_pred = 0
    x_pred = 0
    v_pred = 0
    t_pred = 0

    v_target_list = []
    v_pred_list = []
    x_target_list = []
    x_pred_list = []
    y_target_list = []
    y_pred_list = []
    t_target_list = []
    t_pred_list = []
    
    acc_list = []
    steer_list = [] 

    mpc = MPC()
    a_control = 0
    steer_control = 0
    index = 0
    time_sim = 0

    while index < 200:
      a_control, steer_control = mpc.opt(x_target, y_target, v_target, t_target, x_pred, y_pred, v_pred, t_pred, dt, lf, a_control, steer_control)

      acc_list.append(a_control)
      steer_list.append(steer_control)

      v_pred += a_control*dt[0]

      t_pred += v_pred*dt[0]*steer_control/lf

      x_pred += v_pred*dt[0]*np.cos(t_pred)
      y_pred += v_pred*dt[0]*np.sin(t_pred)

      time_sim += dt[0]

      print ('Index: ' +str(index))
      print ('Vehicle acc: ' + str(acc_list[-1]))
      print ('Vehicle steer: ' + str(np.rad2deg(steer_list[-1])))
      print ('Speed: ' + str(v_pred))
      print ('Yaw: ' + str(np.rad2deg(t_pred)))
      print ('X, Y: ' + str([x_pred, y_pred]))
      print ('Time: ' + str(time_sim))

      v_target_list.append(v_target[0])
      v_pred_list.append(v_pred)
      x_target_list.append(x_target[0])
      x_pred_list.append(x_pred)
      y_target_list.append(y_target[0])
      y_pred_list.append(y_pred)
      t_target_list.append(t_target[0])
      t_pred_list.append(t_pred)

      index += 1
      if index == 30:
         x_target = 0*np.ones(size)
         y_target = 0*np.ones(size)
         t_target = np.deg2rad(45+180)*np.ones(size)
      if index > 30 and (x_pred_list[-1]**2 + y_pred_list[-1]**2) < 10**2:
          break
      if index == 200:
          break

    plt.subplot(231)
    plt.plot(v_target_list, 'blue')
    plt.plot(v_pred_list, 'red', marker='x')
    plt.plot(acc_list, 'green', marker='x')
    plt.xlabel("time (s)")
    plt.ylabel("deg")
    plt.title('Vehicle speed')

    plt.subplot(232)
    plt.plot(np.rad2deg(t_target_list), 'blue')
    plt.plot(np.rad2deg(t_pred_list), 'red', marker='x')
    plt.plot(np.rad2deg(steer_list), 'green', marker='x')
    plt.xlabel("time (s)")
    plt.ylabel("deg")
    plt.title('Yaw')

    plt.subplot(233)
    plt.plot(acc_list, 'blue')
    plt.xlabel("time (s)")
    plt.ylabel("m/ss")
    plt.title('Acc vehicle')

    plt.subplot(234)
    plt.plot(x_target_list, 'blue')
    plt.plot(x_pred_list, 'red', marker='x')
    plt.xlabel("time (s)")
    plt.ylabel("m")
    plt.title('X')

    plt.subplot(235)
    plt.plot(y_target_list, 'blue')
    plt.plot(y_pred_list, 'red', marker='x')
    plt.xlabel("time (s)")
    plt.ylabel("m")
    plt.title('Y')

    plt.subplot(236)
    plt.plot(x_pred_list, y_pred_list, 'blue')
    plt.xlabel("time (s)")
    plt.ylabel("rad")
    plt.title('Steer')

    plt.tight_layout()
    plt.show()


