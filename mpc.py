from transforms3d.euler import euler2mat, mat2euler
from scipy import optimize
import numpy as np
from warnings import filterwarnings
from sympy import *
import matplotlib.pyplot as plt
from sympy.geometry import *
import model_carla as model


class MPC:
    def __init__(self):
       print ('MPC process' )

    def fun(self, x, *args):
      v_pred = args[0]
      v_target = args[1]
      t_pred = args[2]
      t_target = args[3]
      y_pred = args[4]
      y_target = args[5]
      x_pred = args[6]
      x_target = args[7]
      dt = args[8]
      lf = args[9]
      N = args[10]

      vt_0 = v_pred
      tt_0 = t_pred
      xt_0 = x_pred
      yt_0 = y_pred

      vt = [[vt_0 + x[0]*dt[0]]]
      for x_elem, dt_elem in zip(x[1:N], dt[1:]):
        vt = np.append(vt, [[vt[-1][0] + x_elem*dt_elem]], axis=0)

      ev = [[vt[0][0] - v_target[0]]]
      for v_target_elem, vt_elem in zip(v_target[1:], vt[1:]):
          ev = np.append(ev, [[vt_elem[0] - v_target_elem]], axis=0)

      tt = [[tt_0 + vt_0*dt[0]*x[4]/lf]]
      for x_elem, dt_elem, vt_elem in zip(x[N+1:], dt[1:], vt):
          tt = np.append(tt, [[tt[-1][0] + vt_elem[0]*dt_elem*x_elem/lf]], axis=0)

      et = [[tt[0][0] - t_target[0]]]
      for t_target_elem, tt_elem in zip(t_target[1:], tt[1:]):
          et = np.append(et, [[tt_elem[0] - t_target_elem]], axis=0)

      xt = [[xt_0 + vt_0*dt[0]*np.cos(tt_0)]]
      for dt_elem, vt_elem, tt_elem in zip(dt[1:], vt, tt):
        xt = np.append(xt, [[xt[-1][0] + vt_elem[0]*dt_elem*np.cos(tt_elem[0])]], axis=0)

      ex = [[xt[0][0] - x_target[0]]]
      for x_target_elem, xt_elem in zip(x_target[1:], xt[1:]):
          ex = np.append(ex, [[xt_elem[0] - x_target_elem]], axis=0)

      yt = [[yt_0 + vt_0*dt[0]*np.sin(tt_0)]]
      for dt_elem, vt_elem, tt_elem in zip(dt[1:], vt, tt):
        yt = np.append(yt, [[yt[-1][0] + vt_elem[0]*dt_elem*np.sin(tt_elem[0])]], axis=0)

      ey = [[yt[0][0] - y_target[0]]]
      for y_target_elem, yt_elem in zip(y_target[1:], yt[1:]):
          ey = np.append(ey, [[yt_elem[0] - y_target_elem]], axis=0)

      ctet = [[y_target[0] - (yt[0][0] + vt[0][0]*np.sin(et[0][0])*dt[0])]]
      for y_target_elem, yt_elem, vt_elem, et_elem, dt_elem in zip(y_target[1:], yt[1:], vt[1:], et[1:], dt[1:]):
          ctet = np.append(ctet, [[y_target_elem - (yt_elem + vt_elem*np.sin(et_elem)*dt_elem)]])

      error = 0
      for ctet_elem in zip(ctet):
          error += 10000*np.linalg.norm(ctet_elem[0])

      for et_elem in zip(et):
          error += 1000*np.linalg.norm(et_elem[0])

      for ev_elem in zip(ev):
          error += 100*np.linalg.norm(ev_elem[0])

      #for ex_elem in zip(ex):
      #    error += 1000*np.linalg.norm(ex_elem[0])

      #for ey_elem in zip(ey):
      #    error += 1000*np.linalg.norm(ey_elem[0])
      
      for x_elem in x:
          error += 1*np.linalg.norm(x_elem)

      for x_elem_post, x_elem_ant in zip(x[1:N], x[:N-1]):
          error += 10*np.linalg.norm(x_elem_ant - x_elem_post)

      for x_elem_post, x_elem_ant in zip(x[N-1:], x[N:]):
          error += 10*np.linalg.norm(x_elem_ant - x_elem_post)

      return error 

    def fun_1(self, x, *args):
      v_pred = args[0]
      t_pred = args[1]
      y_pred = args[2]
      x_pred = args[3]

      dt = args[4]
      lf = args[5]
      N = args[6]

      poly = args[7]
      coefficients = args[8]

      x_state_list = [[x_pred, y_pred, t_pred, v_pred]]

      if N == 6:
        x_state_list = np.append(x_state_list, [model.move_with_acc(np.asarray(x_state_list[-1]), dt[0], np.asarray([0, x[6], x[0]]), lf)], axis = 0)
        x_state_list = np.append(x_state_list, [model.move_with_acc(np.asarray(x_state_list[-1]), dt[1], np.asarray([0, x[7], x[1]]), lf)], axis = 0)
        x_state_list = np.append(x_state_list, [model.move_with_acc(np.asarray(x_state_list[-1]), dt[2], np.asarray([0, x[8], x[2]]), lf)], axis = 0)
        x_state_list = np.append(x_state_list, [model.move_with_acc(np.asarray(x_state_list[-1]), dt[3], np.asarray([0, x[9], x[3]]), lf)], axis = 0)
        x_state_list = np.append(x_state_list, [model.move_with_acc(np.asarray(x_state_list[-1]), dt[4], np.asarray([0, x[10], x[4]]), lf)], axis = 0)
        x_state_list = np.append(x_state_list, [model.move_with_acc(np.asarray(x_state_list[-1]), dt[5], np.asarray([0, x[11], x[5]]), lf)], axis = 0)
      elif N == 7:
        x_state_list = np.append(x_state_list, [model.move_with_acc(np.asarray(x_state_list[-1]), dt[0], np.asarray([0, x[7], x[0]]), lf)], axis = 0)
        x_state_list = np.append(x_state_list, [model.move_with_acc(np.asarray(x_state_list[-1]), dt[1], np.asarray([0, x[8], x[1]]), lf)], axis = 0)
        x_state_list = np.append(x_state_list, [model.move_with_acc(np.asarray(x_state_list[-1]), dt[2], np.asarray([0, x[9], x[2]]), lf)], axis = 0)
        x_state_list = np.append(x_state_list, [model.move_with_acc(np.asarray(x_state_list[-1]), dt[3], np.asarray([0, x[10], x[3]]), lf)], axis = 0)
        x_state_list = np.append(x_state_list, [model.move_with_acc(np.asarray(x_state_list[-1]), dt[4], np.asarray([0, x[11], x[4]]), lf)], axis = 0)
        x_state_list = np.append(x_state_list, [model.move_with_acc(np.asarray(x_state_list[-1]), dt[5], np.asarray([0, x[12], x[5]]), lf)], axis = 0)
        x_state_list = np.append(x_state_list, [model.move_with_acc(np.asarray(x_state_list[-1]), dt[6], np.asarray([0, x[13], x[6]]), lf)], axis = 0)
      elif N == 10:
        x_state_list = np.append(x_state_list, [model.move_with_acc(np.asarray(x_state_list[-1]), dt[0], np.asarray([0, x[10], x[0]]), lf)], axis = 0)
        x_state_list = np.append(x_state_list, [model.move_with_acc(np.asarray(x_state_list[-1]), dt[1], np.asarray([0, x[11], x[1]]), lf)], axis = 0)
        x_state_list = np.append(x_state_list, [model.move_with_acc(np.asarray(x_state_list[-1]), dt[2], np.asarray([0, x[12], x[2]]), lf)], axis = 0)
        x_state_list = np.append(x_state_list, [model.move_with_acc(np.asarray(x_state_list[-1]), dt[3], np.asarray([0, x[13], x[3]]), lf)], axis = 0)
        x_state_list = np.append(x_state_list, [model.move_with_acc(np.asarray(x_state_list[-1]), dt[4], np.asarray([0, x[14], x[4]]), lf)], axis = 0)
        x_state_list = np.append(x_state_list, [model.move_with_acc(np.asarray(x_state_list[-1]), dt[5], np.asarray([0, x[15], x[5]]), lf)], axis = 0)
        x_state_list = np.append(x_state_list, [model.move_with_acc(np.asarray(x_state_list[-1]), dt[6], np.asarray([0, x[16], x[6]]), lf)], axis = 0)
        x_state_list = np.append(x_state_list, [model.move_with_acc(np.asarray(x_state_list[-1]), dt[6], np.asarray([0, x[17], x[7]]), lf)], axis = 0)
        x_state_list = np.append(x_state_list, [model.move_with_acc(np.asarray(x_state_list[-1]), dt[6], np.asarray([0, x[18], x[8]]), lf)], axis = 0)
        x_state_list = np.append(x_state_list, [model.move_with_acc(np.asarray(x_state_list[-1]), dt[6], np.asarray([0, x[19], x[9]]), lf)], axis = 0)

      #print ('State')
      #print (x_state_list[1], x_state_list[2], x_state_list[3])

      x_target = x_state_list[1:,0]
      y_target = poly(x_target)
      coefficients_der = np.polyder(coefficients)
      t_target = np.poly1d(coefficients_der)(x_target)
      
      #print ('Target')
      #print ([x_target, y_target, t_target])
      v_target = 3.2*np.ones(N) # m/s

      ctet = [[x_state_list[1,1] - y_target[0]]]
      for y_target_elem, yt_elem in zip(y_target[1:], x_state_list[2:,1]):
          ctet = np.append(ctet, [[yt_elem - y_target_elem]])

      et = [[x_state_list[1,2] - t_target[0]]]
      for t_target_elem, tt_elem in zip(t_target[1:], x_state_list[2:,2]):
          et = np.append(et, [[tt_elem - t_target_elem]], axis=0)

      ev = [[x_state_list[1,3] - v_target[0]]]
      for v_target_elem, vt_elem in zip(v_target[1:], x_state_list[2:,3]):
          ev = np.append(ev, [[vt_elem - v_target_elem]], axis=0)

      error = 0
      for ctet_elem in zip(ctet):
          error += 2000*np.linalg.norm(ctet_elem[0])

      for et_elem in zip(et):
          error += 1000*np.linalg.norm(et_elem[0])

      for ev_elem in zip(ev):
          error += 7000*np.linalg.norm(ev_elem[0])

      for x_elem in x:
          error += 1*np.linalg.norm(x_elem)

      for x_elem_post, x_elem_ant in zip(x[1:N], x[:N-1]):
          error += 30*np.linalg.norm(x_elem_ant - x_elem_post)

      for x_elem_post, x_elem_ant in zip(x[N-1:], x[N:]):
          error += 1000*np.linalg.norm(x_elem_ant - x_elem_post)

      return error 
            
    def opt(self, x_pred, y_pred, t_pred, v_pred, dt, wheelbase, acc, steer, N, poly, coefficients):

      acc_max = 2 
      acc_min = -2
      steer_max = np.deg2rad(10)
      steer_min = -np.deg2rad(10)

      acc_min_array = acc_min*np.ones(N)
      acc_max_array = acc_max*np.ones(N)
      steer_min_array = steer_min*np.ones(N)
      steer_max_array = steer_max*np.ones(N)

      if N == 6:
        cons = ({'type': 'ineq', 'fun': lambda x: x[0] + acc_max},
                {'type': 'ineq', 'fun': lambda x: -x[0] + acc_max},
                {'type': 'ineq', 'fun': lambda x: x[1] + acc_max},
                {'type': 'ineq', 'fun': lambda x: -x[1] + acc_max},
                {'type': 'ineq', 'fun': lambda x: x[2] + acc_max},
                {'type': 'ineq', 'fun': lambda x: -x[2] + acc_max},
                {'type': 'ineq', 'fun': lambda x: x[3] + acc_max},
                {'type': 'ineq', 'fun': lambda x: -x[3] + acc_max},
                {'type': 'ineq', 'fun': lambda x: x[4] + acc_max},
                {'type': 'ineq', 'fun': lambda x: -x[4] + acc_max},
                {'type': 'ineq', 'fun': lambda x: x[5] + acc_max},
                {'type': 'ineq', 'fun': lambda x: -x[5] + acc_max},
                {'type': 'ineq', 'fun': lambda x: x[6] + steer_max},
                {'type': 'ineq', 'fun': lambda x: -x[6] + steer_max},
                {'type': 'ineq', 'fun': lambda x: x[7] + steer_max},
                {'type': 'ineq', 'fun': lambda x: -x[7] + steer_max},
                {'type': 'ineq', 'fun': lambda x: x[8] + steer_max},
                {'type': 'ineq', 'fun': lambda x: -x[8] + steer_max},
                {'type': 'ineq', 'fun': lambda x: x[9] + steer_max},
                {'type': 'ineq', 'fun': lambda x: -x[9] + steer_max},
                {'type': 'ineq', 'fun': lambda x: x[10] + steer_max},
                {'type': 'ineq', 'fun': lambda x: -x[10] + steer_max},
                {'type': 'ineq', 'fun': lambda x: x[11] + steer_max},
                {'type': 'ineq', 'fun': lambda x: -x[11] + steer_max})
      elif N == 7:
        cons = ({'type': 'ineq', 'fun': lambda x: x[0] + acc_max},
                {'type': 'ineq', 'fun': lambda x: -x[0] + acc_max},
                {'type': 'ineq', 'fun': lambda x: x[1] + acc_max},
                {'type': 'ineq', 'fun': lambda x: -x[1] + acc_max},
                {'type': 'ineq', 'fun': lambda x: x[2] + acc_max},
                {'type': 'ineq', 'fun': lambda x: -x[2] + acc_max},
                {'type': 'ineq', 'fun': lambda x: x[3] + acc_max},
                {'type': 'ineq', 'fun': lambda x: -x[3] + acc_max},
                {'type': 'ineq', 'fun': lambda x: x[4] + acc_max},
                {'type': 'ineq', 'fun': lambda x: -x[4] + acc_max},
                {'type': 'ineq', 'fun': lambda x: x[5] + acc_max},
                {'type': 'ineq', 'fun': lambda x: -x[5] + acc_max},
                {'type': 'ineq', 'fun': lambda x: x[6] + acc_max},
                {'type': 'ineq', 'fun': lambda x: -x[6] + acc_max},
                {'type': 'ineq', 'fun': lambda x: x[7] + steer_max},
                {'type': 'ineq', 'fun': lambda x: -x[7] + steer_max},
                {'type': 'ineq', 'fun': lambda x: x[8] + steer_max},
                {'type': 'ineq', 'fun': lambda x: -x[8] + steer_max},
                {'type': 'ineq', 'fun': lambda x: x[9] + steer_max},
                {'type': 'ineq', 'fun': lambda x: -x[9] + steer_max},
                {'type': 'ineq', 'fun': lambda x: x[10] + steer_max},
                {'type': 'ineq', 'fun': lambda x: -x[10] + steer_max},
                {'type': 'ineq', 'fun': lambda x: x[11] + steer_max},
                {'type': 'ineq', 'fun': lambda x: -x[11] + steer_max},
                {'type': 'ineq', 'fun': lambda x: x[12] + steer_max},
                {'type': 'ineq', 'fun': lambda x: -x[12] + steer_max},
                {'type': 'ineq', 'fun': lambda x: x[13] + steer_max},
                {'type': 'ineq', 'fun': lambda x: -x[13] + steer_max})
      elif N == 10:
        cons = ({'type': 'ineq', 'fun': lambda x: x[0] + acc_max},
                {'type': 'ineq', 'fun': lambda x: -x[0] + acc_max},
                {'type': 'ineq', 'fun': lambda x: x[1] + acc_max},
                {'type': 'ineq', 'fun': lambda x: -x[1] + acc_max},
                {'type': 'ineq', 'fun': lambda x: x[2] + acc_max},
                {'type': 'ineq', 'fun': lambda x: -x[2] + acc_max},
                {'type': 'ineq', 'fun': lambda x: x[3] + acc_max},
                {'type': 'ineq', 'fun': lambda x: -x[3] + acc_max},
                {'type': 'ineq', 'fun': lambda x: x[4] + acc_max},
                {'type': 'ineq', 'fun': lambda x: -x[4] + acc_max},
                {'type': 'ineq', 'fun': lambda x: x[5] + acc_max},
                {'type': 'ineq', 'fun': lambda x: -x[5] + acc_max},
                {'type': 'ineq', 'fun': lambda x: x[6] + acc_max},
                {'type': 'ineq', 'fun': lambda x: -x[6] + acc_max},
                {'type': 'ineq', 'fun': lambda x: x[7] + acc_max},
                {'type': 'ineq', 'fun': lambda x: -x[7] + acc_max},
                {'type': 'ineq', 'fun': lambda x: x[8] + acc_max},
                {'type': 'ineq', 'fun': lambda x: -x[8] + acc_max},
                {'type': 'ineq', 'fun': lambda x: x[9] + acc_max},
                {'type': 'ineq', 'fun': lambda x: -x[9] + acc_max},
                {'type': 'ineq', 'fun': lambda x: x[10] + steer_max},
                {'type': 'ineq', 'fun': lambda x: -x[10] + steer_max},
                {'type': 'ineq', 'fun': lambda x: x[11] + steer_max},
                {'type': 'ineq', 'fun': lambda x: -x[11] + steer_max},
                {'type': 'ineq', 'fun': lambda x: x[12] + steer_max},
                {'type': 'ineq', 'fun': lambda x: -x[12] + steer_max},
                {'type': 'ineq', 'fun': lambda x: x[13] + steer_max},
                {'type': 'ineq', 'fun': lambda x: -x[13] + steer_max},
                {'type': 'ineq', 'fun': lambda x: x[14] + steer_max},
                {'type': 'ineq', 'fun': lambda x: -x[14] + steer_max},
                {'type': 'ineq', 'fun': lambda x: x[15] + steer_max},
                {'type': 'ineq', 'fun': lambda x: -x[15] + steer_max},
                {'type': 'ineq', 'fun': lambda x: x[16] + steer_max},
                {'type': 'ineq', 'fun': lambda x: -x[16] + steer_max},
                {'type': 'ineq', 'fun': lambda x: x[17] + steer_max},
                {'type': 'ineq', 'fun': lambda x: -x[17] + steer_max},
                {'type': 'ineq', 'fun': lambda x: x[18] + steer_max},
                {'type': 'ineq', 'fun': lambda x: -x[18] + steer_max},
                {'type': 'ineq', 'fun': lambda x: x[19] + steer_max},
                {'type': 'ineq', 'fun': lambda x: -x[19] + steer_max})

      bounds_min = np.append(acc_min_array, steer_min_array)
      bounds_max = np.append(acc_max_array, steer_max_array)

      #bounds = Bounds(bounds_min, bounds_max)

      x0 = acc*np.ones(N)
      x0 = np.append(x0, steer*np.ones(N))

      options = {'maxiter': 10000, 'disp': True}

      #sol = optimize.minimize(self.fun_1, args=(v_pred, t_pred, y_pred, x_pred, dt, wheelbase, N, poly, coefficients), x0=x0, method='trust-constr', bounds=bounds, options=options)
      sol = optimize.minimize(self.fun_1, args=(v_pred, t_pred, y_pred, x_pred, dt, wheelbase, N, poly, coefficients), x0=x0, method='COBYLA', options=options, constraints=cons)
      #sol = optimize.minimize(self.fun_1, args=(v_pred, t_pred, y_pred, x_pred, dt, wheelbase, N, poly, coefficients), x0=x0, method='SLSQP', options=options, constraints=cons)
      # sol: at_0, at_1, at_2, at_3, steert_0, steert_1, steert_2, steert_3
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


