import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from transforms3d.euler import euler2mat, mat2euler
import transforms3d.derivations.eulerangles
import json
from scipy.signal import medfilt, medfilt2d, wiener, spline_filter, cubic
import mpc
import utils
import scipy
import sys, time
import numpy as np
import model_carla as model


from scipy import interpolate

if __name__== "__main__":

    if len(sys.argv) == 2:


        x_carla = []
        x_ref = []
        x_model_estimated = []

        y_carla = []
        y_ref = []
        y_model_estimated = []

        yaw_ref = []
        yaw_model_estimated = []

        wheelbase = []
        x_speed_veh = []
        veh_speed_model_estimated = []
        steer = []

        time_list = []
        x_state_vector_ref = []
        x_acc_veh = []
        throttle = []
        brake = []
        yaw_ref = []

        file_p = open(sys.argv[1], 'r')
        line = file_p.readline()

        #prius_parameters = {'steer': 30, 'wheelbase': 2.22455}
        prius_parameters = {'steer': np.deg2rad(80), 'wheelbase': 2.22}
        time_increment = 0

        while len(line) != 0: 
            data = eval(line)

            if x_ref == [] or ((data['x'] - x_ref[-1])**2 + (data['y'] - y_ref[-1])**2) > 0.5**2:
              x_ref.append(data['x'])
              y_ref.append(data['y'])
              yaw_ref.append(np.deg2rad(data['yaw']))
              
              steer.append(data['steer'])
              x_acc_veh.append(data['x_acc']) 
              
              wheelbase.append(data['wheelbase'])
              time_increment += data['deltaseconds']
              x_speed_veh.append(data['lon_speed'])
              throttle.append(data['throttle'])
              brake.append(-data['brake'])
              
              if time_list == []:
                  time_list.append(time_increment)
              else:
                  time_list.append(time_list[-1]+time_increment)
              
              wheelbase_para = 1.6 
              if x_state_vector_ref == []:
                 x_state_vector_ref.append([x_ref[-1], y_ref[-1], yaw_ref[-1], x_speed_veh[-1]]) 
                 print (x_state_vector_ref)
              else: 
                 u = [steer[-1], x_acc_veh[-1]/10]
                 x_state_vector_ref.append(model.move_with_acc(x_state_vector_ref[-1], time_increment, u, wheelbase_para))
              
              # Representation
              x_model_estimated.append(x_state_vector_ref[-1][0])
              y_model_estimated.append(x_state_vector_ref[-1][1])
              yaw_model_estimated.append(model.normalize_angle(x_state_vector_ref[-1][2]))
              veh_speed_model_estimated.append(x_state_vector_ref[-1][3])

              time_increment = 0

            else:
              time_increment += data['deltaseconds'] 

            line = file_p.readline()


        mpc_flag = True

        if mpc_flag == False:
           plt.subplot(221)
           plt.plot(x_ref, y_ref)
           plt.plot(x_model_estimated, y_model_estimated, 'red')
           plt.subplot(222)
           plt.plot(yaw_ref)
           plt.plot(yaw_model_estimated, 'red')
           plt.title('Yaw')
           plt.subplot(223)
           plt.plot(x_acc_veh)
           plt.title('XAcc')
           plt.subplot(224)
           plt.plot(np.rad2deg(steer))
           plt.title('Steer')
           plt.show()
           exit()

        mpc = mpc.MPC()

        x_mpc = []
        y_mpc = []
        yaw_mpc = []
        speed_mpc = []
        steer_mpc = []
        a_mpc = []
        time_mpc = []
        
        acc_control = 0
        steer_control = 0

        start_index = 0 
        end_index = len(x_ref) - 50
        print ('Lenght: ' + str(len(x_ref)))

        x_state_vector_transf = []

        curve_fitting_flag = False
        if curve_fitting_flag == True:
          i = start_index
          angles = []
          axs = []
          ays = []
          while i < end_index:
            print ('--------------------')
            print ('Index: ' + str(i))
            N = 30

            points_ref = np.transpose(np.array([(x_ref[i:i+N]), (y_ref[i:i+N]), (np.zeros(N))])) # Reference points

            # Poly fit
            coefficients_x = np.polyfit(points_ref[:,0], points_ref[:,1], N-1)
            coefficients_y = np.polyfit(points_ref[:,1], points_ref[:,0], N-1)
            y_poly = np.poly1d(coefficients_x)
            x_poly = np.poly1d(coefficients_y)
            points_y_poly = np.transpose(np.array([(points_ref[:,0]), (y_poly(points_ref[:,0])), (np.zeros(N))])) # Reference points
            points_x_poly = np.transpose(np.array([(x_poly(points_ref[:,1])), (points_ref[:,1]), (np.zeros(N))])) # Reference points

            offset = 20
            # Data ploting
            plt.subplot(1,3,1)
            plt.plot(points_ref[:,0], points_ref[:,1], 'yellow', marker='^', linestyle='None')
            plt.plot(points_ref[0,0], points_ref[0,1], 'blue', marker='x')
            plt.plot(points_ref[:,0], y_poly(points_ref[:,0]), 'green')
            plt.ylim([points_ref[0,1]-offset, points_ref[0,1]+offset])
            plt.xlim([points_ref[0,0]-offset, points_ref[0,0]+offset])
            plt.title('Y_poly')
            x_distance_error = 0
            y_distance_error = 0
            for a, b in zip(points_ref, points_y_poly):
                x_distance_error += np.linalg.norm(a-b)
            for a, b in zip(points_ref, points_x_poly):
                y_distance_error += np.linalg.norm(a-b)

            print (x_distance_error)
            print (y_distance_error)
            plt.text(points_ref[0,0]-offset, points_ref[0,1]-3, str(x_distance_error), fontsize=12)

            plt.subplot(1,3,2)
            plt.plot(points_ref[:,0], points_ref[:,1], 'yellow', marker='^', linestyle='None')
            plt.plot(points_ref[0,0], points_ref[0,1], 'blue', marker='x')
            plt.plot(x_poly(points_ref[:,1]), points_ref[:,1], 'green')
            plt.ylim([points_ref[0,1]-offset, points_ref[0,1]+offset])
            plt.xlim([points_ref[0,0]-offset, points_ref[0,0]+offset])
            plt.text(points_ref[0,0]+3, points_ref[0,1]+3, str(y_distance_error), fontsize=12)
            plt.text(points_ref[0,0]+1, points_ref[0,1]+1, str(yaw_ref[i]), fontsize=12)
            plt.title('X_poly')

            plt.savefig('mpc_results/mpc_result_'+str(i)+'.png')
            plt.close()
            #plt.show()
            i += 1


        x_state_vector = []
        if mpc_flag == True:
          i = start_index
          while i < end_index:
                print ('--------------------')
                print ('Index: ' + str(i))


                v_target = 3.5 # Speed reference
                if i > 100:
                    v_target = 8;

                N = 20
                dt_mpc = 0.2

                points_ref = np.transpose(np.array([(x_ref[i:i+N]), (y_ref[i:i+N]), (np.zeros(N))])) # Reference points

                # Poly fit
                coefficients_x = np.polyfit(points_ref[:,0], points_ref[:,1], N-1)
                coefficients_y = np.polyfit(points_ref[:,1], points_ref[:,0], N-1)
                y_poly = np.poly1d(coefficients_x)
                x_poly = np.poly1d(coefficients_y)
                points_y_poly = np.transpose(np.array([(points_ref[:,0]), (y_poly(points_ref[:,0])), (np.zeros(N))])) # Reference points
                points_x_poly = np.transpose(np.array([(x_poly(points_ref[:,1])), (points_ref[:,1]), (np.zeros(N))])) # Reference points

                x_distance_error = 0
                y_distance_error = 0

                for a, b in zip(points_ref, points_y_poly):
                    x_distance_error += np.linalg.norm(a-b)
                for a, b in zip(points_ref, points_x_poly):
                    y_distance_error += np.linalg.norm(a-b)

                # Controller
                if x_state_vector == []:
                    print ('Inicialitation x_state_vector_transf')
                    x_state_vector = np.asarray([[x_ref[start_index], y_ref[start_index], yaw_ref[start_index], 0]])
                    print (x_state_vector)

                # Reference path is evaluated and MPC is called. 
                if x_distance_error < y_distance_error:
                  sol_mpc = mpc.opt(x_state_vector[-1], dt_mpc*np.ones(N), wheelbase_para, acc_control, steer_control, N, y_poly, coefficients_x, 'y_poly', v_target)
                else:
                  sol_mpc = mpc.opt(x_state_vector[-1], dt_mpc*np.ones(N), wheelbase_para, acc_control, steer_control, N, x_poly, coefficients_y, 'x_poly', v_target)

                # Car simulator.
                print ('Initial state')
                aux = [x_state_vector[-1]]
                sol_index = 0
                for acc_elem, steer_elem in zip(sol_mpc[:N], sol_mpc[N:]):
                  u = [steer_elem, acc_elem]
                  print ('Soluciones de control ' + str(sol_index) + ': ' + str(u))
                  aux.append(model.move_with_acc(np.asarray(aux[-1]), dt_mpc, u, wheelbase_para, debug=True))
                  print ('Estado ' + str(sol_index) + ': ' + str(aux[-1]))
                  sol_index += 1
                aux = np.asarray(aux)
                
                # Car simulator.
                u = [sol_mpc[N], sol_mpc[0]]
                x_state_vector = np.append(x_state_vector, [model.move_with_acc(np.asarray(x_state_vector[-1]), dt_mpc, u, wheelbase_para, debug=True)], axis=0)
                print ('Contro Action:')
                print (u)
                print ('x_state_vector:')
                print(x_state_vector[-1])

                # Data representation
                car_vector = {'x': x_state_vector[-1,0], 'y': x_state_vector[-1,1], 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': x_state_vector[-1,2]} 
                car_matrix = utils.vector_to_matrix_pose(car_vector)

                ref_vector = {'x': x_ref[i], 'y': y_ref[i], 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': yaw_ref[i]} 
                ref_matrix = utils.vector_to_matrix_pose(ref_vector)

                diff_matrix = np.dot(np.linalg.inv(ref_matrix), car_matrix)
                diff_vector = utils.matrix_to_vector_pose(diff_matrix)

                radio = 2

                ax = plt.gca()
                plt.cla()
                plt.plot(points_ref[:,0], points_ref[:,1], 'yellow', marker='^', linestyle='None')
                plt.plot(points_ref[0,0], points_ref[0,1], 'green', marker='x', linestyle='None')
                if x_distance_error < y_distance_error:
                  plt.plot(points_ref[:,0], y_poly(points_ref[:,0]), 'green')
                else:
                  plt.plot(x_poly(points_ref[:,1]), points_ref[:,1], 'green')
                plt.plot(aux[1:,0], aux[1:,1], 'blue', marker='o')

                circulo = matplotlib.patches.Circle(xy=(x_state_vector[-1,0], x_state_vector[-1,1]), radius=radio, fill=None)
                ax.add_patch(circulo)

                offset = 10
                plt.ylim([points_ref[0,1]-offset, points_ref[0,1]+offset])
                plt.xlim([points_ref[0,0]-offset, points_ref[0,0]+offset])
                plt.text(points_ref[0,0]+2, points_ref[0,1]+3, 'x_poly: ' + str(y_distance_error), fontsize=12)
                plt.text(points_ref[0,0]+2, points_ref[0,1]+4, 'y_poly: ' + str(x_distance_error), fontsize=12)
                if x_distance_error < y_distance_error:
                  plt.text(points_ref[0,0]+2, points_ref[0,1]+2, 'y_poly', fontsize=12)
                else:
                  plt.text(points_ref[0,0]+2, points_ref[0,1]+2, 'x_poly', fontsize=12)
                plt.text(points_ref[0,0]+1, points_ref[0,1]+1, str(yaw_ref[i]), fontsize=12)
                plt.text(points_ref[0,0]-offset, points_ref[0,1]-2, 'x: ' + str(diff_vector['x']) + ' y: ' + str(diff_vector['y']) + ' yaw: ' + str(diff_vector['yaw']), fontsize=12)
                plt.text(points_ref[0,0]-offset, points_ref[0,1]-4, 'x: ' + str(car_vector['x']) + ' y: ' + str(car_vector['y']) + ' yaw: ' + str(car_vector['yaw']), fontsize=12)
                plt.text(points_ref[0,0]-offset, points_ref[0,1]-6, 'x: ' + str(ref_vector['x']) + ' y: ' + str(ref_vector['y']) + ' yaw: ' + str(ref_vector['yaw']), fontsize=12)
                plt.text(points_ref[0,0]-offset, points_ref[0,1]-8, 'dx: ' + str(car_vector['x'] - ref_vector['x']) + ' dy: ' + str(car_vector['y'] - ref_vector['y']) + ' ayaw: ' + str(car_vector['yaw'] - ref_vector['yaw']), fontsize=12)
 
                plt.savefig('mpc_results/mpc_result_'+str(i)+'.png')
                #plt.close()
                #plt.show()


                if time_mpc == []:
                    time_mpc.append(time_list[i])
                else:
                    time_mpc.append(time_mpc[-1]+dt_mpc)

                
                if diff_vector['x'] > 15:
                    exit()
                print (diff_vector)
                x_distance_error = x_state_vector[-1,0]-points_ref[0,0]
                y_distance_error = x_state_vector[-1,1]-points_ref[0,1]
                print ('X Distance error: ' + str(x_distance_error))
                print ('Y Distance error: ' + str(y_distance_error))
                for x_elem, y_elem in zip(points_ref[:,0], points_ref[:,1]):
                  if ((x_elem - x_state_vector[-1,0])**2 + (y_elem - x_state_vector[-1,1])**2) > radio**2:
                    if diff_vector['x'] > 0: # Referencia detras de coche
                      i += 1
                    else:
                        break
                  else:
                    i += 1
                    break


                x_mpc.append(x_state_vector[-1,0])
                y_mpc.append(x_state_vector[-1,1])
                yaw_mpc.append(x_state_vector[-1,2])
                speed_mpc.append(x_state_vector[-1,3])
                steer_mpc.append(u[0])
                a_mpc.append(u[1])

        plt.subplot(331)
        plt.plot(time_mpc, np.rad2deg(yaw_mpc), 'green')
        plt.xlabel("time (s)")
        plt.ylabel("deg")
        plt.title('Yaw')

        plt.subplot(332)
        plt.plot(time_mpc, x_mpc, 'green')
        plt.xlabel("time (s)")
        plt.ylabel("deg")
        plt.title('X')

        plt.subplot(333)
        plt.plot(time_mpc, y_mpc, 'green')
        plt.xlabel("time (s)")
        plt.ylabel("deg")
        plt.title('Y')

        plt.subplot(334)
        plt.plot(time_mpc, np.rad2deg(steer_mpc), 'green')
        plt.ylabel("deg")
        plt.xlabel("s")
        plt.title('steer')

        plt.subplot(336)
        plt.plot(time_mpc, speed_mpc, 'green')
        plt.ylabel("m/s")
        plt.xlabel("s")
        plt.title('speed')

        plt.subplot(337)
        plt.plot(time_mpc, a_mpc, 'green')
        plt.xlabel("time (s)")
        plt.ylabel("m/ss")
        plt.title('X acc vehicle')

        plt.subplot(339)
        plt.plot(x_ref[start_index:end_index], y_ref[start_index:end_index], 'blue')
        plt.plot(x_mpc, y_mpc, 'green')
        plt.xlabel("m")
        plt.ylabel("m")
        plt.title('Trajectory')

        plt.savefig('mpc_results/mpc_result_000.png')
        plt.show()

