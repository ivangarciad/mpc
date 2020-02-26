import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from transforms3d.euler import euler2mat, mat2euler
import transforms3d.derivations.eulerangles
import json
from scipy.signal import medfilt, medfilt2d, wiener, spline_filter, cubic
import model_carla as model
import mpc
import scipy
import sys, time
import numpy as np
import bezier

from scipy import interpolate


font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

if __name__== "__main__":

    if len(sys.argv) == 2:

        yaw = []
        yaw_estimate = []

        roll = []
        roll_estimate = []

        pitch = []
        pitch_estimate = []

        x_carla = []
        x_ekf = []
        x_estimate = []

        y_carla = []
        y_ekf = []
        y_estimate = []

        wheelbase = []
        deltaseconds = []
        lon_speed = []
        lon_speed_estimated = []
        steer = []

        z_angle = []

        time_list = []
        x_state_vector = []
        x_acc_veh = []
        throttle = []
        brake = []
        yaw_ref = []

        file_p = open(sys.argv[1], 'r')
        line = file_p.readline()

        #prius_parameters = {'steer': 30, 'wheelbase': 2.22455}
        prius_parameters = {'steer': np.deg2rad(80), 'wheelbase': 2.22}

        while len(line) != 0: 
            data = eval(line)

            yaw.append(np.deg2rad(data['yaw']))
            roll.append(data['roll'])
            pitch.append(data['pitch'])
            x_ekf.append(data['x'])
            y_ekf.append(data['y'])
            wheelbase.append(data['wheelbase']-prius_parameters['wheelbase'])
            deltaseconds.append(data['deltaseconds'])
            lon_speed.append(data['lon_speed'])
            #steer.append(data['steer']*np.deg2rad(15))
            steer.append(data['steer']*np.deg2rad(57))
            z_angle.append(data['z_speed']*deltaseconds[-1])
            x_acc_veh.append(data['x_acc']) 
            throttle.append(data['throttle'])
            brake.append(-data['brake'])
            yaw_ref.append(-data['yaw_ref'])

            if time_list == []:
                time_list.append(deltaseconds[-1])
            else:
                time_list.append(time_list[-1]+deltaseconds[-1])

            line = file_p.readline()
            wheelbase_para = .55 
            wheelbase_para = 1.7 
            if x_state_vector == []:
               x_state_vector.append([x_ekf[0], y_ekf[0], yaw[0], lon_speed[0]]) 
               print (x_state_vector)
            else: 
               x_state_vector.append(model.move_with_acc_new(x_state_vector[-1], deltaseconds[-1], [0, steer[-1], x_acc_veh[-1]/3.5], wheelbase_para))

            x_estimate.append(x_state_vector[-1][0])
            y_estimate.append(x_state_vector[-1][1])
            yaw_estimate.append(model.normalize_angle(x_state_vector[-1][2]))
            lon_speed_estimated.append(x_state_vector[-1][3])

        

        mpc = mpc.MPC()
        

        x_mpc = []
        y_mpc = []
        yaw_mpc = []
        speed_mpc = []
        steer_mpc = []
        a_mpc = []
        time_mpc = []
        
        
        a_control = 0
        steer_control = 0
        aux_index = 0


        first_time = True
        count = 0
        mpc_flag = False
        start_index = 1050 
        end_index = len(x_ekf) - 300
        print ('Lenght: ' + str(len(x_ekf)))

        x_state_vector = [x_ekf[start_index], y_ekf[start_index], yaw[start_index], 0]

        if mpc_flag == True:
          i = start_index
          index = 0
          while i < end_index:
                print ('--------------------')
                print ('Index: ' + str(i))
                print ('Current position: ' + str([x_ekf[i], y_ekf[i], 0]))
                print ('Yaw: ' + str(yaw[i]))
                print ('Yaw State Vector: ' + str(x_state_vector[2]))
                start = time.time()
                
                points_new = np.transpose(np.array([(x_ekf[i:i+60]), (y_ekf[i:i+60]), (np.zeros(60))]))

                nodes1 = np.asfortranarray([points_new[:,0], points_new[:,1]])
                curve1 = bezier.Curve(nodes1, degree=59)

                s_vals = np.linspace(0.0, 1, 10000)
                points = curve1.evaluate_multi(s_vals)
                x_bezier = points[0, :]
                y_bezier = points[1, :]

                points_bezier = np.transpose(np.asarray([(points[0,:]), (points[1,:]), (np.zeros(len(points[0,:])))]))

                rotation = transforms3d.euler.euler2mat(0, 0, -x_state_vector[2], axes='sxyz')
                rotation_inv = transforms3d.euler.euler2mat(0, 0, x_state_vector[2], axes='sxyz')

                points_new_transf = []
                for point in points_bezier:
                  points_new_transf.append(np.dot(rotation, point))

                points_new_transf = np.asarray(points_new_transf)

                x_state_vector_transf = np.copy(x_state_vector)
                pose = np.dot(rotation, [x_state_vector[0], x_state_vector[1], 0])
                x_state_vector_transf[0] = pose[0]
                x_state_vector_transf[1] = pose[1]
                x_state_vector_transf[2] = 0
                x_state_vector_transf[3] = x_state_vector[3]

                N = 6
                print (N)
                samples_per_T = 4

                x_ref = points_new[0:50,0]
                y_ref = points_new[0:50,1]

                point_ref_transf = np.asarray([np.dot(rotation, [x_ref[0], y_ref[0], 0])]) 
                for x, y in zip(x_ref[1:], y_ref[1:,]):
                  point_ref_transf = np.append(point_ref_transf, [np.dot(rotation, [x, y, 0])], axis=0)

                
                coefficients_x = np.polyfit(points_new_transf[:,0], points_new_transf[:,1], 10)
                y_poly = np.poly1d(coefficients_x)
                coefficients = coefficients_x
                poly = y_poly

                x_ref_transf = point_ref_transf[:,0]
                y_ref_transf = poly(point_ref_transf[:,0])

                point_ref = np.asarray([np.dot(rotation_inv, [x_ref_transf[0], y_ref_transf[0], 0])]) 
                for x, y in zip(x_ref_transf[1:], y_ref_transf[1:,]):
                  point_ref = np.append(point_ref, [np.dot(rotation_inv, [x, y, 0])], axis=0)

                sol_mpc = mpc.opt(x_state_vector_transf[0], x_state_vector_transf[1], x_state_vector_transf[2], x_state_vector_transf[3], samples_per_T*np.asarray(deltaseconds[i:i+N]), wheelbase_para, a_control, steer_control, N, poly, coefficients)
                end = time.time()
                print('Elapsed time: ' + str(end - start))

                x_state_list = [[x_state_vector[0], x_state_vector[1], x_state_vector[2], x_state_vector[3]]]
                print ('x_state_list: ' + str(x_state_list[-1]))
                for acc_elem, steer_elem in zip(sol_mpc[:N], sol_mpc[N:]):
                    print ('Soluciones de control: ' + str([acc_elem, steer_elem]))
                    x_state_list.append(model.move_with_acc(x_state_list[-1], samples_per_T*deltaseconds[i], [0, steer_elem, acc_elem], wheelbase_para, debug=True))
                    print ('Estado: ' + str(x_state_list[-1]))
                
                x_state = np.asarray(x_state_list)

                point_state_transf = np.asarray([np.dot(rotation, [x_state[0,0], x_state[0,1], 0])]) 
                for x, y in zip(x_state[1:,0], x_state[1:,1]):
                  point_state_transf = np.append(point_state_transf, [np.dot(rotation, [x, y, 0])], axis=0)

                x_state_transf = point_state_transf[:,0]
                y_state_transf = poly(point_state_transf[:,0])

                point_state = np.asarray([np.dot(rotation_inv, [x_state_transf[0], y_state_transf[0], 0])]) 
                for x, y in zip(x_state_transf[1:], y_state_transf[1:,]):
                  point_state = np.append(point_state, [np.dot(rotation_inv, [x, y, 0])], axis=0)

                #plt.plot(points[:,0], points[:,1], 'orange', marker='^')
                ax = curve1.plot(1000)
                ax.plot(x_bezier, y_bezier, marker='x', linewidth=4, linestyle='None', color='green')
                plt.plot(points_new[:,0], points_new[:,1], 'yellow', marker='^', linestyle='None')
 
                #plt.plot(point_ref[:,0], point_ref[:,1], 'black', marker='^')
                plt.plot(point_state[1:,0], point_state[1:,1], 'red', marker='x')
 
                plt.plot(x_state[0,0], x_state[0,1], 'green', marker='o')
                plt.plot(x_state[1:,0], x_state[1:,1], 'blue', marker='o')
                plt.plot([x_state[0,0], x_state[0,0]+0.5*np.cos(x_state[0,2])], [x_state[0,1], x_state[0,1]+0.5*np.sin(x_state[0,2])], 'green')
 
                plt.savefig('mpc_restuls/mpc_result_'+str(i)+'.png')
                plt.close()
                #plt.show()

                a_control = sol_mpc[0]
                steer_control = sol_mpc[N]

                #x_state_vector = model.move_with_acc(x_state_vector, deltaseconds[i], [0, steer_control, a_control], wheelbase_para) # Se define el periodo de muestreo del sistema con 20ms=0.01s. Simulacion del simulador o coche real
                x_state_vector = x_state[1]

                print ('x_state_vector: ' + str(x_state_vector))
                

                if time_mpc == []:
                    time_mpc.append(time_list[i])
                else:
                    time_mpc.append(samples_per_T*deltaseconds[i]+time_mpc[-1])

                distance_list = []
                for x_elem, y_elem in zip(points_new[:,0], points_new[:,1]):
                  distance = (x_elem - x_state_vector[0])**2 + (y_elem - x_state_vector[1])**2
                  distance_list.append(distance)  

                index = 0
                for dist_ant, dist, dist_sig in zip(distance_list[0:], distance_list[1:], distance_list[2:]):
                    index += 1
                    if dist_ant < dist and dist < 1.0:
                        i += index
                        print ([index, dist_ant])
                        break

                #print (distance_list)
                #if index == len(distance_list):
                #    i += index
                #    same_ref = False

                x_mpc.append(x_state_vector[0])
                y_mpc.append(x_state_vector[1])
                yaw_mpc.append(x_state_vector[2])
                speed_mpc.append(x_state_vector[3])
                steer_mpc.append(steer_control)
                a_mpc.append(a_control)

        plt.subplot(331)
        plt.plot(time_list[start_index:end_index], np.rad2deg(yaw[start_index:end_index]), 'blue')
        plt.plot(time_list[start_index:end_index], np.rad2deg(yaw_estimate[start_index:end_index]), 'red')
        if mpc_flag == True:
           plt.plot(time_mpc, np.rad2deg(yaw_mpc), 'green')
        plt.xlabel("time (s)")
        plt.ylabel("deg")
        plt.title('Yaw')

        plt.subplot(332)
        plt.plot(time_list, x_ekf, 'blue')
        plt.plot(time_list, x_estimate, 'red')
        if mpc_flag == True:
          plt.plot(time_mpc, x_mpc, 'green')
        plt.xlabel("time (s)")
        plt.ylabel("deg")
        plt.title('X')

        plt.subplot(333)
        plt.plot(time_list, y_ekf, 'blue')
        plt.plot(time_list, y_estimate, 'red')
        if mpc_flag == True:
          plt.plot(time_mpc, y_mpc, 'green')
        plt.xlabel("time (s)")
        plt.ylabel("deg")
        plt.title('Y')

        plt.subplot(334)
        plt.plot(time_list[start_index:end_index], np.rad2deg(steer[start_index:end_index]), 'blue')
        plt.plot(time_list[start_index:end_index], np.rad2deg(z_angle[start_index:end_index]), 'red')
        if mpc_flag == True:
          plt.plot(time_mpc, np.rad2deg(steer_mpc), 'green')
        plt.ylabel("deg")
        plt.title('steer')

        #start_index = 0
        plt.subplot(335)
        plt.plot(x_ekf[start_index:end_index], y_ekf[start_index:end_index], 'blue', marker='^')
        plt.plot(x_estimate[start_index:end_index], y_estimate[start_index:end_index], 'red')
        if mpc_flag == True:
          plt.plot(x_mpc, y_mpc, 'green', marker='x')
        plt.xlabel("m")
        plt.ylabel("m")
        plt.title('Trajectory')

        plt.subplot(336)
        plt.plot(time_list[start_index:end_index], lon_speed[start_index:end_index], 'blue')
        plt.plot(time_list[start_index:end_index], lon_speed_estimated[start_index:end_index], 'red')
        if mpc_flag == True:
          plt.plot(time_mpc, speed_mpc, 'green')
        plt.ylabel("m/s")
        plt.title('speed')

        plt.subplot(337)
        plt.plot(time_list[start_index:end_index], x_acc_veh[start_index:end_index], 'blue')
        if mpc_flag == True:
          plt.plot(time_mpc, a_mpc, 'green')
        plt.xlabel("time (s)")
        plt.ylabel("m/ss")
        plt.title('X acc vehicle')

        plt.subplot(338)
        plt.plot(time_list, throttle, 'blue')
        plt.plot(time_list, brake, 'red')
        plt.xlabel("time (s)")
        plt.ylabel("m/ss")
        plt.title('Throttle/Brake')

        if yaw_ref != []:
          plt.subplot(339)
          plt.plot(time_list, np.rad2deg(yaw_ref), 'blue')
          plt.xlabel("time (s)")
          plt.ylabel("m/ss")
          plt.title('Yaw ref')

        #plt.tight_layout()
        plt.savefig('mpc_restuls/mpc_result_000.png')
        plt.show()

