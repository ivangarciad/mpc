import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from transforms3d.euler import euler2mat, mat2euler
import transforms3d.derivations.eulerangles
import json
from scipy.signal import medfilt, medfilt2d, wiener, spline_filter, cubic
import mpc
import scipy
import sys, time
import numpy as np
import model_carla as model
import bezier
import sympy

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
        deltaseconds = []
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

        while len(line) != 0: 
            data = eval(line)

            if x_ref == [] or ((data['x'] - x_ref[-1])**2 + (data['y'] - y_ref[-1])**2) > 0.2:
              x_ref.append(data['x'])
              y_ref.append(data['y'])
              yaw_ref.append(data['yaw'])
              
              steer.append(data['steer'])
              x_acc_veh.append(data['x_acc']) 
              
              wheelbase.append(data['wheelbase'])
              deltaseconds.append(data['deltaseconds'])
              x_speed_veh.append(data['lon_speed'])
              throttle.append(data['throttle'])
              brake.append(-data['brake'])
              
              if time_list == []:
                  time_list.append(deltaseconds[-1])
              else:
                  time_list.append(time_list[-1]+deltaseconds[-1])
              
              wheelbase_para = 1.7 
              if x_state_vector_ref == []:
                 x_state_vector_ref.append([x_ref[0], y_ref[0], yaw_ref[0], x_speed_veh[0]]) 
                 print (x_state_vector_ref)
              else: 
                 u = [steer[-1], x_acc_veh[-1]/3.5]
                 x_state_vector_ref.append(model.move_with_acc(x_state_vector_ref[-1], deltaseconds[-1], u, wheelbase_para))
              
              # Representation
              x_model_estimated.append(x_state_vector_ref[-1][0])
              y_model_estimated.append(x_state_vector_ref[-1][1])
              yaw_model_estimated.append(model.normalize_angle(x_state_vector_ref[-1][2]))
              veh_speed_model_estimated.append(x_state_vector_ref[-1][3])

            line = file_p.readline()


        mpc_flag = True

        if mpc_flag == False:
           plt.subplot(221)
           plt.plot(x_ref, y_ref, marker='x')
           plt.subplot(222)
           plt.plot(yaw_ref)
           plt.title('Yaw')
           plt.subplot(223)
           plt.plot(x_acc_veh)
           plt.title('XAcc')
           plt.subplot(224)
           plt.plot(steer)
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
        end_index = len(x_ref) - 500
        print ('Lenght: ' + str(len(x_ref)))
        print ('dt: ' + str(deltaseconds[0]))

        curve_fitting_flag = True
        x_state_vector_transf = []
        angle_to_ratate = 0
        if curve_fitting_flag == True:
          i = start_index
          angles = []
          axs = []
          ays = []
          while i < end_index:
            print ('--------------------')
            print ('Index: ' + str(i))
            N = 30

            angle_to_ratate = model.normalize_angle(yaw_ref[i] - yaw_ref[i+1])
            print (angle_to_ratate)
            rotation = transforms3d.euler.euler2mat(0, 0, angle_to_ratate, axes='sxyz')

            points_ref = np.transpose(np.array([(x_ref[i:i+N]), (y_ref[i:i+N]), (np.zeros(N))])) # Reference points

            nodes1 = np.asfortranarray([points_ref[:,0], points_ref[:,1]])
            curve1 = bezier.Curve(nodes1, degree=N-1)

            print ('Points bezier')
            point1 = np.asfortranarray([[points_ref[N-2,0]],[points_ref[N-2,1]]])
            s = curve1.locate(point1)
            if s != None:
              print ('S value: ' +str(s))
              print (curve1.evaluate(s))
              tangent = curve1.evaluate_hodograph(s)
              print (tangent)
              axs.append(tangent[0])
              ays.append(tangent[1])
              angle = np.arctan2(tangent[1],tangent[0])
              angles.append(angle)
              print ('Angulo: ' + str(angle))

            if N <5:
               y_bezier_new = []
               bezier_sympy = curve1.implicitize()
               print ('--------------')
               x, y = sympy.symbols('x y')
               for point_x in points_ref[:,0]:
                 result = bezier_sympy.subs(x, point_x)  
                 y_value = sympy.solve(result, y)
                 print (y_value)
                 print ([point_x, y_value[0]])
                 y_bezier_new.append(sympy.re(y_value[0]))
               
            s_vals = np.linspace(0.0, 1, N)
            bezier_points = curve1.evaluate_multi(s_vals)
            x_bezier = bezier_points[0,:]
            y_bezier = bezier_points[1,:]

            # Poly fit
            points_ref_transf = []
            for point in points_ref:
                points_ref_transf.append(np.dot(rotation, point))

            points_ref_transf = np.asarray(points_ref_transf)
            print ('References at i: ' + str(points_ref_transf[0]))
            print (len(points_ref_transf))

            coefficients_x = np.polyfit(points_ref[:,0], points_ref[:,1], N-1)
            coefficients_y = np.polyfit(points_ref[:,1], points_ref[:,0], N-1)
            y_poly = np.poly1d(coefficients_x)
            x_poly = np.poly1d(coefficients_y)
            points_y_poly = []
            for x, y in zip(points_ref[:,0], y_poly(points_ref[:,0])): 
                points_y_poly.append([x, y, 0])
            points_x_poly = []
            for x, y in zip(x_poly(points_ref[:,1]), points_ref[:,1]): 
                points_x_poly.append([x, y, 0])

            points_y_poly = np.asarray(points_y_poly)
            points_x_poly = np.asarray(points_x_poly)

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

            if N < 4:
              plt.subplot(1,3,3)
              plt.plot(points_ref[:,0], points_ref[:,1], 'yellow', marker='^', linestyle='None')
              plt.plot(points_ref[0,0], points_ref[0,1], 'blue', marker='x')
              plt.plot(x_bezier, y_bezier, 'blue', marker='^')
              plt.plot(points_ref[:,0], y_bezier_new, 'red')
              plt.ylim([y_bezier[0]-offset, y_bezier[0]+offset])
              plt.xlim([x_bezier[0]-offset, x_bezier[0]+offset])
              plt.title('Bezier')

            plt.savefig('mpc_results/mpc_result_'+str(i)+'.png')
            plt.close()
            #plt.show()
            i += 1

          plt.figure(1)
          plt.plot(angles)
          plt.figure(2)
          plt.plot(axs)
          plt.figure(3)
          plt.plot(ays)
          plt.show()

        exit()
        if mpc_flag == True:
          i = start_index
          while i < end_index:
                print ('--------------------')
                print ('Index: ' + str(i))
                N = 6

                x_state_vector_ref = [x_ref[i], y_ref[i], yaw_ref[i], 0]

                rotation = transforms3d.euler.euler2mat(0, 0, -x_state_vector_ref[2], axes='sxyz')
                rotation_inv = transforms3d.euler.euler2mat(0, 0, x_state_vector_ref[2], axes='sxyz')

                points_ref = np.transpose(np.array([(x_ref[i:i+20*N]), (y_ref[i:i+20*N]), (np.zeros(20*N))])) # Reference points

                points_ref_transf = []
                for point in points_ref:
                    points_ref_transf.append(np.dot(rotation, point))

                if x_state_vector_transf == []:
                    print ('Inicialitation x_state_vector_transf')
                    x_state_vector_transf = np.asarray([[points_ref_transf[0][0], points_ref_transf[0][1], points_ref_transf[0][2], 0]])

                points_ref_transf = np.asarray(points_ref_transf)
                print ('References at i: ' + str(points_ref_transf[-1]))

                coefficients_x = np.polyfit(points_ref_transf[:,0], points_ref_transf[:,1], 15)
                y_poly = np.poly1d(coefficients_x)

                # Reference path is evaluated and MPC is called. 
                time_scale = 10
                dt_mpc = 0.2
                sol_mpc = mpc.opt(x_state_vector_transf[-1], points_ref_transf[:,0], dt_mpc*np.ones(N), wheelbase_para, acc_control, steer_control, N, y_poly, coefficients_x)

                print ('Initial state')
                if len(x_state_vector_transf) >= 2:
                  aux = [x_state_vector_transf[-2,:]]
                else:
                  aux = [x_state_vector_transf[-1,:]]

                # Car simulator.
                sol_index = 0
                for acc_elem, steer_elem in zip(sol_mpc[:N], sol_mpc[N:]):
                    u = [steer_elem, acc_elem]
                    print ('Soluciones de control ' + str(sol_index) + ': ' + str(u))
                    aux.append(model.move_with_acc(np.asarray(aux[-1]), time_scale*deltaseconds[i], u, wheelbase_para, debug=True))
                    print ('Estado ' + str(sol_index) + ': ' + str(aux[-1]))
                    sol_index += 1
                
                acc_control = sol_mpc[0]
                steer_control = sol_mpc[N]

                # Car simulator.
                u = [steer_control, acc_control]
                x_state_vector_transf = np.append(x_state_vector_transf, [model.move_with_acc(np.asarray(x_state_vector_transf[-1,:]), time_scale*deltaseconds[i], u, wheelbase_para, debug=True)], axis=0)
                print ('Contro Action:')
                print (u)
                print ('x_state_vector:')
                print(x_state_vector_transf[-1])

                aux = np.asarray(aux)
                radio = 2

                ax = plt.gca()
                plt.cla()
                plt.plot(points_ref_transf[:,0], points_ref_transf[:,1], 'yellow', marker='^', linestyle='None')
                plt.plot(points_ref_transf[:,0], y_poly(points_ref_transf[:,0]), 'green')
                #plt.plot(x_state_vector_transf[-2,0], x_state_vector_transf[-2,1], 'green', marker='o')
                #plt.plot(aux[:,0], aux[:,1], 'blue', marker='o')

                #circulo = matplotlib.patches.Circle(xy=(x_state_vector_transf[-2,0], x_state_vector_transf[-2,1]), radius=radio, fill=None)
                #ax.add_patch(circulo)
                #plt.ylim([5, 40])
 
                plt.savefig('mpc_results/mpc_result_'+str(i)+'.png')
                #plt.close()
                plt.show()


                if time_mpc == []:
                    time_mpc.append(time_list[i])
                else:
                    time_mpc.append(time_scale*deltaseconds[i]+time_mpc[-1])

                for x_elem, y_elem in zip(points_ref_transf[:,0], points_ref_transf[:,1]):
                  if len(x_state_vector_transf) >= 2:
                    distance = (x_elem - x_state_vector_transf[-2,0])**2 + (y_elem - x_state_vector_transf[-2,1])**2
                  else:
                    distance = (x_elem - x_state_vector_transf[-1,0])**2 + (y_elem - x_state_vector_transf[-1,1])**2
                  if distance > radio:
                    i += 1
                  else:
                    i += 1
                    break


                x_mpc.append(x_state_vector_ref[0])
                y_mpc.append(x_state_vector_ref[1])
                yaw_mpc.append(x_state_vector_ref[2])
                speed_mpc.append(x_state_vector_ref[3])
                steer_mpc.append(steer_control)
                a_mpc.append(acc_control)

        plt.subplot(331)
        plt.plot(time_list[start_index:end_index], np.rad2deg(yaw_ref[start_index:end_index]), 'blue')
        plt.plot(time_list[start_index:end_index], np.rad2deg(yaw_model_estimated[start_index:end_index]), 'red')
        if mpc_flag == True:
           plt.plot(time_mpc, np.rad2deg(yaw_mpc), 'green')
        plt.xlabel("time (s)")
        plt.ylabel("deg")
        plt.title('Yaw')

        plt.subplot(332)
        plt.plot(time_list, x_ref, 'blue')
        plt.plot(time_list, x_model_estimated, 'red')
        if mpc_flag == True:
          plt.plot(time_mpc, x_mpc, 'green')
        plt.xlabel("time (s)")
        plt.ylabel("deg")
        plt.title('X')

        plt.subplot(333)
        plt.plot(time_list, y_ref, 'blue')
        plt.plot(time_list, y_model_estimated, 'red')
        if mpc_flag == True:
          plt.plot(time_mpc, y_mpc, 'green')
        plt.xlabel("time (s)")
        plt.ylabel("deg")
        plt.title('Y')

        plt.subplot(334)
        plt.plot(time_list[start_index:end_index], np.rad2deg(steer[start_index:end_index]), 'blue')
        if mpc_flag == True:
          plt.plot(time_mpc, np.rad2deg(steer_mpc), 'green')
        plt.ylabel("deg")
        plt.title('steer')


        plt.subplot(336)
        plt.plot(time_list[start_index:end_index], x_speed_veh[start_index:end_index], 'blue')
        plt.plot(time_list[start_index:end_index], veh_speed_model_estimated[start_index:end_index], 'red')
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

        #start_index = 0
        plt.subplot(339)
        plt.plot(x_ref[start_index:end_index], y_ref[start_index:end_index], 'blue')
        plt.plot(x_model_estimated[start_index:end_index], y_model_estimated[start_index:end_index], 'red')
        if mpc_flag == True:
          plt.plot(x_mpc, y_mpc, 'green', marker='x')
        plt.xlabel("m")
        plt.ylabel("m")
        plt.title('Trajectory')
        plt.xlim([-10,-8])

        #plt.tight_layout()
        #plt.savefig('mpc_restuls/mpc_result_000.png')
        plt.show()

