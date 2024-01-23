"""PSO for tuning PID params for quadrotor(DJI-Mavic2pro)
using a neural network arch. 
1. Create Randomized params for PIDs(x, y, z, roll, pitch, yaw):
2. Collect performance fitness from given PID parms
3. Use NN to tune params
"""


# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
###IMPORTS#######################################
from optparse import Option
from re import X
import sys, os
import csv, math
import struct

from pyswarms.backend import Swarm
from controller import Robot, Supervisor
from cmath import pi

try: 
    import numpy as np
    import pandas as pd
    import pyswarms as ps
    import pyswarms.backend as psb
    #import tensorflow as tf
    #from keras import layers, models
    from csv import DictWriter
    from scipy.signal import find_peaks
    from scipy import integrate
    from simple_pid import PID
    from sklearn.metrics import mean_squared_error
    from pyswarms.backend.swarms import Swarm
except ImportError:
    sys.exit("Warning: 'numpy' module not found.")
#################################################




#CrazyFlie Supervisor for PSO
class CrazyFlieSuper(Supervisor):
    #constructor
    def __init__(self, time_step):
        Supervisor.__init__(self)

        try:
            self.time_step = time_step
        except:
            self.time_step = int(self.getBasicTimeStep())
        print("Supervisor using timestep: {}".format(self.time_step))
        
        #dictionary to files
        self.files_dict = {}
        
        #array to meas. avg/std across particles
        self.avgstd_arr = np.array([])
        #array to meas. fitness avg/std across particles
        self.avgfit_arr = np.array([])
        
        #list of particles
        self.particles = []

        # Get and enable devices
        #include emitter and receiver to get comms with Supervisor
        self.super_rcvr = self.getDevice("super_rcvr")
        self.super_rcvr.enable(self.time_step)
        self.super_emtr = self.getDevice("super_emtr")

        self.waypoint_idx = 0
        self.waypoints = []
        self.curr_waypnt = [0, 0, 0]
        self.next_waypnt = [0, 0, 0]
        self.yaw_feat_exp = 0
        self.iter_num = 0
        self.cleared_crs = False
        

    #if acc_idx == size of arays, 
    #then-->roll the arrays to left and write val to last index 
    def rollDimArr(self, dim_arr, time_arr):
        dim_arr = np.roll(dim_arr, -1)
        time_arr = np.roll(time_arr, -1)


    #find way to return a single list of times from each feature
    def processStateFileFeats(self, input_arr, timestep):
        trials_list = []
        times_list = []
        start_time = 0.032####needs to be changed if world timestep is chnged
        i = 0
        while i < len(timestep):
            feat_trial = []
            time_trial = []
            if timestep[i] == start_time:
                feat_trial.clear()
                feat_trial.append(input_arr[i])
                time_trial.append(timestep[i])
                i += 1
            elif timestep[i] != start_time:
                j = i
                if j < len(timestep):
                    while timestep[j] != start_time:
                        feat_trial.append(input_arr[j])
                        time_trial.append(timestep[j])
                        j += 1
                        if j > len(timestep) - 1:
                            break
                    i = j
                    trials_list.append(feat_trial)
                    times_list.append(time_trial)
                else:
                    print("end of arr")
        return trials_list, times_list
    

    #generate trajectory path to use for error comparison w/ reference
    def generatePitchRef(self, init_waypt, tgt_waypt, calcd_timestep, num_of_waypoints=0):
        calcd_dist = np.sqrt(((tgt_waypt[1] - init_waypt[1]) ** 2) + ((tgt_waypt[0] - init_waypt[0]) ** 2))
       #print("creating {} waypoints for pitch-MSE".format(num_of_waypoints))

        delta_x = tgt_waypt[0] - init_waypt[0]
        delta_y = tgt_waypt[1] - init_waypt[0]
    
        # Calculate the step size for each waypoint
        if num_of_waypoints > 1:
            step_x = delta_x / (num_of_waypoints - 1) 
            step_y = delta_y / (num_of_waypoints - 1)
        else:
            step_x = 0
            step_y = 0
    
        # Generate the list of waypoints
        waypoints = []
        for i in range(int(num_of_waypoints)):
            x = init_waypt[0] + i * step_x
            y = init_waypt[1] + i * step_y
            waypoints.append((x, y))
        return waypoints


    #display error between flight trajectory and 
    #get true-trajectory distance from center
    def getFlightMSE(self, x_true, y_true, x_traj, y_traj):
        true_dist = np.sqrt(np.square(x_true) + np.square(y_true))
        true_dist_avg = np.round(np.mean(true_dist), 4)
        #print("true_dist_avg: {}".format(true_dist_avg))
        true_dist = np.full(len(x_traj), true_dist_avg)
        #print(true_dist)
        #print(len(true_dist))
        #get flight-trajectory distance from cetner
        flgt_dist = np.sqrt(np.square(x_traj) + np.square(y_traj))
        #print("flgt_dist avg: {}".format(np.mean(flgt_dist)))
        #print(len(flgt_dist))
        #collect mse(true vs. flight)
        if np.mean(flgt_dist) < float(1):
            flgt_MSE = float(99)
            flgt_SSE = float(99)
        else:
            flgt_MSE = mean_squared_error(true_dist, flgt_dist)
            flgt_SSE = np.log10(np.sum((flgt_dist - true_dist)**2))
        #print("MSE: {:.4f}".format(flgt_MSE))
        print("log10-SSE: {:.4f}".format(flgt_SSE))
        return flgt_SSE
        #return flgt_MSE


    #collect exp-traj with ref-traj during pitch-tuning
    #meas. MSE(exp-traj, ref-traj)
    def getFitnessFromTrialsMSE(self,
                                x_trials, 
                                x_trials_times, 
                                y_trials,
                                y_trials_times,
                                feat_exp=(float(0), float(0)),
                                calcd_time_step=0.032):
        fitness_avg_arr = []
        #print("# of trials: {}".format(len(x_trials)))
        for i in range(len(x_trials)):
            x_trial = x_trials[i]
            x_trial_time = x_trials_times[i]
            y_trial = y_trials[i]
            y_trial_time = y_trials_times[i]
         #   print("len of x_trial: {}".format(len(x_trial)))
         #   print("len of y_trial: {}".format(len(y_trial)))
        
            if len(x_trial) < 1000:
                fitness_avg_arr.append(float(99))
          #      print("MSE: {:.4f}".format(float(99)))
            else:
                init_waypt = (x_trial[0], y_trial[0])
                step = y_trial_time[1] - y_trial_time[0]
                pitch_refpath = np.array(self.generatePitchRef(init_waypt, 
                                                               feat_exp, 
                                                               step, 
                                                               num_of_waypoints=len(x_trial)))
        
                x_ref = np.array([])
                y_ref = np.array([])
                for i, ref_pt in enumerate(pitch_refpath):
                    x_ref = np.append(x_ref, ref_pt[0])
                    y_ref = np.append(y_ref, ref_pt[1])
        
                trajMSE = self.getFlightMSE(x_ref, y_ref, x_trial, y_trial)
                final_trial_time = y_trial_time[len(y_trial_time) - 1]
                print("final trial time: {}".format(final_trial_time))
                fitness_avg_arr.append(trajMSE + np.log10(final_trial_time))#MSE plus time
        
        return np.mean(np.array([fitness_avg_arr]))


    #return reference yaw angle of yaw-tuning
    def getYawChgAngle(self, curr_waypt, tgt_waypt, curr_yaw):
        #calculate yaw_setpoint for target waypoint
        #This will be in ]-pi;pi]
        yaw_chg = np.arctan2(tgt_waypt[1] - curr_waypt[1],
                                tgt_waypt[0] - curr_waypt[0])
        #print("calcd yaw chg: {:.4f}".format(yaw_chg))

        # This is now in ]-2pi;2pi[
        angle_left = yaw_chg - curr_yaw
        #print("angle-left: {:.4f}".format(angle_left))

        # Normalize turn angle to ]-pi;pi]
        angle_left = (angle_left + 2 * np.pi) % (2 * np.pi)
        #print("norm angle-left: {:.4f}".format(angle_left))

        if (angle_left > np.pi):
            angle_left -= 2 * np.pi
            #print("new angle left: {}".format(angle_left))
        return angle_left


    #update optim-file with new waypoint
    def updateOptimFile(self, params, files_dict):
        optim_file = files_dict["optim-file"]
        clearFileIfExists(optim_file)
        with open(optim_file, 'a', newline='', encoding='utf-8') as opt_obj:
            w = csv.writer(opt_obj)
            w.writerows(params.items())
            opt_obj.close()
        print("optim file updated...")
        

    #measure the avg. and std. dev of distribution for params for particles
    def measAvgStdDevforParticles(self):
        print("collecting initial mean and std-dev for params")
        particles_list = []
        parent_file = os.path.dirname(os.getcwd())
        optim_dir = os.path.normpath(os.path.join(parent_file + "\PID_params_exp_PSO"))
        for path, curr_dir, files in os.walk(optim_dir):
            for file in files:
                if file.startswith("optim_file"):
                    particles_list.append(pd.read_csv(optim_dir + "\\" + file, header=None))

        particleDF = pd.DataFrame(columns=particles_list[0][0])
        for i, particle in enumerate(particles_list):
            print("particle no. {}".format(i))
            particle_dict = pd.pivot_table(particle, values=1, columns=[0]).to_dict('records')
            particleDF = pd.concat([particleDF, pd.DataFrame.from_records(particle_dict)])
    
        params_obsvd = ["x_Kp", "x_Ki", "x_Kd",
                        "y_Kp", "y_Ki", "y_Kd",
                        "roll_Kp", "roll_Ki", "roll_Kd",
                        "pitch_Kp", "pitch_Ki", "pitch_Kd",
                        "yaw_Kp", "yaw_Ki", "yaw_Kd",
                        "throttle_Kp", "throttle_Ki", "throttle_Kd"]

        performance_dict = {}
        for param in params_obsvd:
            param_arr = np.array(particleDF[param])
            param_mean = np.mean(param_arr)
            param_std = np.std(param_arr)
            performance_dict[param] = (param_mean, param_std)

        return performance_dict


    #write avg-std over particles and avg fit per iteration
    def writeParticleFiles(self):
        #save fitness array
        fitness_file = self.files_dict['state-filedir'] + r"\fitness_list.npy"
        clearFileIfExists(fitness_file)
        with open(fitness_file, 'wb') as fit_file:
            np.save(fit_file, self.avgfit_arr)
            fit_file.close()

        #save array with dictionaries of param avg/std-dev over iterations
        param_perf_file = self.files_dict['state-filedir'] + r"\param_perf_list.npy"
        clearFileIfExists(param_perf_file)
        with open(param_perf_file, 'wb') as perf_file:
            np.save(perf_file, self.avgstd_arr)
            perf_file.close()

    #writes state for mavic2pro/robot-node
    def writeMavicState(self, input_arr, filename):
        field_names = ['x_pos', 'y_pos', 'z_pos', 
                       'roll_rot', 'pitch_rot', 'yaw_rot', 
                       'x_vel', 'y_vel', 'alt_vel',
                       'x_acc', 'y_acc', 'alt_acc',
                       'roll_vel', 'pitch_vel', 'yaw_vel',
                       'timestep',
                       'front_left_motor', 'front_right_motor',
                       'rear_left_motor', 'rear_right_motor']

        csv_dict = {'x_pos': input_arr[0],
                    'y_pos': input_arr[1],
                    'z_pos': input_arr[2],
                    'roll_rot': input_arr[3],
                    'pitch_rot': input_arr[4],
                    'yaw_rot': input_arr[5],
                    'x_vel': input_arr[6],
                    'y_vel': input_arr[7],
                    'alt_vel': input_arr[8],
                    'x_acc': input_arr[9],
                    'y_acc': input_arr[10],
                    'alt_acc': input_arr[11],
                    'roll_vel': input_arr[12],
                    'pitch_vel': input_arr[13],
                    'yaw_vel': input_arr[14],
                    'timestep': input_arr[15],
                    'front_left_motor': input_arr[16],
                    'front_right_motor': input_arr[17],
                    'rear_left_motor': input_arr[18],
                    'rear_right_motor': input_arr[19]}

        with open(filename, 'a', newline='', encoding='utf-8') as f_obj:
            dictWriter_obj = DictWriter(f_obj, fieldnames=field_names)
            try:
                dictWriter_obj.writerow(csv_dict)
            except:
                print("sim crashed on writing state...")
                self.cleared_crs = True
            f_obj.close()


    #write PID and DOF-inputs for mavic2pro/robot-node
    def writePIDandInputs(self, input_arr, filename):
        field_names = ['xposPID', 'yposPID',
                       'rollPID', 'pitchPID', 
                       'yawPID', 'throttlePID',
                       'roll_input', 'pitch_input', 
                       'yaw_input', 'vertical_input',
                       'diff_altitude', 'clampd_diff_altitude']

        csv_dict = {'xposPID': input_arr[0],
                    'yposPID': input_arr[1],
                    'rollPID': input_arr[2],
                    'pitchPID': input_arr[3],
                    'yawPID': input_arr[4],
                    'throttlePID':input_arr[5],
                    'roll_input': input_arr[6],
                    'pitch_input': input_arr[7],
                    'yaw_input': input_arr[8],
                    'vertical_input': input_arr[9],
                    'diff_altitude': input_arr[10],
                    'clampd_diff_altitude': input_arr[11]}

        with open(filename, 'a') as f_obj:
            dictWriter_obj = DictWriter(f_obj, fieldnames=field_names)
            try:
                dictWriter_obj.writerow(csv_dict)
            except:
                print("sim crashed on writing PID/input...")
                self.cleared_crs = True
            f_obj.close()

            
    #return reference yaw angle of yaw-tuning
    def getYawChgAngle(self, curr_waypt, tgt_waypt, curr_yaw):
        #calculate yaw_setpoint for target waypoint
        #This will be in ]-pi;pi]
        yaw_chg = np.arctan2(tgt_waypt[1] - curr_waypt[1],
                                tgt_waypt[0] - curr_waypt[0])
        #print("calcd yaw chg: {:.4f}".format(yaw_chg))

        # This is now in ]-2pi;2pi[
        angle_left = yaw_chg - curr_yaw
        #print("angle-left: {:.4f}".format(angle_left))

        # Normalize turn angle to ]-pi;pi]
        angle_left = (angle_left + 2 * np.pi) % (2 * np.pi)
        #print("norm angle-left: {:.4f}".format(angle_left))

        if (angle_left > np.pi):
            angle_left -= 2 * np.pi
            #print("new angle left: {}".format(angle_left))
        return angle_left
    
    
    #collect overshoot and adjust-time fitness for oscillating DOFs
    def getOvshootFitness(self,
                          trials, 
                          trials_times, 
                          calcd_time_step=0.032, 
                          feat_exp=0.001, 
                          time_exp=3.0,
                          ovshoot_exp = 0.02,
                          exp_error=0.05):
        fitness_avg_arr = []
        #print("# of trials: {}".format(len(trials)))
        for i, trial in enumerate(trials):
            trial_time = trials_times[i]
            #print("len of trial: {}".format(len(trial)))
            #print("type of trial: {}".format(type(trial[0])))
            #collect peaks from trial
            peaks, peak_props = find_peaks(trial, height=feat_exp, distance=10)
            print("No. of peaks found: {}".format(len(peaks)))
            
            #for i, peak in enumerate(peaks):
            #    print("peak-{}: {:.4f}".format(i+1, trial[peak]))

            peak_bal_val = 0    #value at suff. steady-state
            peak_time_indx = [] #list for peak times
            peak_cnt = 0    #num of peaks
            peak_sum = 0    #sum of peaks
            #while bal-val is not found and we have not looked at all the peaks
            while peak_bal_val == 0 and peak_cnt < len(peaks):
                peak = peaks[peak_cnt] #get index of where peak is at
                peak_val = trial[peak] #get peak-val
                peak_time = trial_time[peak] #get time at peak
                peak_time_indx.append(peak_time) 
            #    print("peak-{}: {}".format(peak_cnt+1, peak_val))
            #    print("peak-time: {}".format(peak_time))
                peak_cnt+=1
            print("peak times: {}".format(peak_time_indx))

            final_adj_time = 0  #time where drone settles to exp
            dim_arr = np.full((1, 400), 99.9)[0]
            time_arr = np.full((1, 400), 99.9)[0]
            dim_idx = 0
            dim_mean_arr = []
            time_mean_arr = []
            dim_slope_arr = []
                
            #if we have peaktimes
            if len(peak_time_indx) > 0:
                pk_time = peak_time_indx[0] #get init peak time
                idx = peaks[0] #get init idx of peaks
                bal_time = 0 #balance time
                #while our peak pk-time has not overgone final time of trian and adj-time is not found
                while pk_time < trial_time[len(trial_time)-1] and final_adj_time == 0:
                    dim_temp = trial[idx]#get dim
                    time_temp = trial_time[idx] #get time dim
                    #roll dim/time arrs if write at end
                    if dim_idx == 400: 
                        self.rollDimArr(dim_arr, time_arr)
                        dim_arr[-1] = dim_temp
                        time_arr[-1] = time_temp
                        dim_idx = 0
                    #else write dim-val ito array
                    else:
                        dim_arr[dim_idx] = dim_temp
                        time_arr[dim_idx] = time_temp
                        dim_idx+=1

                    dim_mean = np.mean(dim_arr) #dim-mean
                    time_mean = np.mean(time_arr) #time-mean

                    #get slope of dim-arr to verify no major-chg
                    dim_slope = integrate.simpson(dim_arr, time_arr)
                    dim_slope_arr.append(dim_slope)

                    #bool-array for w/in slope bounds
                    flat_slope_bool_a = np.array([np.round(dim_slope, 3) > float(40),
                                                np.round(dim_slope, 3) < float(45)])
                    #if altitude is now stedy and below exp-error
                    if abs(feat_exp - dim_mean) < exp_error and np.all(flat_slope_bool_a):
                        #collect time as final-adj-time
                        final_adj_time = time_temp
                        print("final time: {}".format(final_adj_time))

                    flat_slope_bool_b = np.array([np.round(dim_slope, 3) > float(80),
                                               np.round(dim_slope, 3) < float(85)])    
                    if abs(feat_exp - dim_mean) < 0.5 and np.all(flat_slope_bool_b):
                        final_adj_time = time_temp * float(10)
                        print("final time: {}".format(final_adj_time))


                    dim_mean_arr.append(dim_mean)
                    time_mean_arr.append(time_mean)

                    idx += 1
                    pk_time = trial_time[idx]

                #if final-adj-time is not found
                if final_adj_time == 0:
                    adjust_time = 999
                #adj-time found, collect adjust-time
                else:
                    adjust_time = final_adj_time - peak_time_indx[0]
                overshoot = trial[peaks[0]] - feat_exp
            else:
                #undershot_wgt = abs(np.array(trial).max() - feat_exp)
                adjust_time = 999
                #overshoot = abs(np.array(trial).max() - feat_exp) * float(5) #reward multiplier
                overshoot = 999
            print("adjustment-time: {:.4f}".format(adjust_time))
            print("overshoot: {}".format(overshoot))
            
            fitness = np.log10((adjust_time/time_exp) + 1) + np.log10((overshoot/ovshoot_exp) + 1)
            print("fitness: {}".format(fitness))
            fitness_avg_arr.append(fitness)

        print("average fitness: {}".format(np.mean(np.array(fitness_avg_arr))))
        return np.mean(np.array(fitness_avg_arr))



    
    #collect fitness score from state-records
    def collectFitnessScore(self, files_dict, params, fitness_feats, calcd_time_step=0.032):
        bot_state_dir = files_dict["state-file"]
        #print("bot-state-dir: {}".format(bot_state_dir))
        print("collecting fitness for particles...")
        cols = ['x_pos', 'y_pos', 'z_pos', 
                'roll_rot', 'pitch_rot', 'yaw_rot', 
                'x_vel', 'y_vel', 'alt_vel',
                'x_acc', 'y_acc', 'alt_acc',
                'roll_vel', 'pitch_vel', 'yaw_vel',
                'timestep',
                'front_left_motor', 'front_right_motor',
                'rear_left_motor', 'rear_right_motor']

        bot_stateDF = pd.read_csv(bot_state_dir, names=cols)
        #print(bot_stateDF.shape)
        #print(bot_stateDF.head(15))

        #analyze roll
        print("collecting roll-fitness...")
        roll_rot = np.array(bot_stateDF["roll_rot"])
        timestep = np.array(bot_stateDF["timestep"])
        #print("roll-fitness feat: ({}, {})".format(float(fitness_feats[1][0]), float(fitness_feats[1][1])))
        roll_trials, roll_times = self.processStateFileFeats(roll_rot, timestep=timestep)
        roll_fitness = 0
        #roll_fitness = self. getOvshootFitness(roll_trials,
        #                                       roll_times,
        #                                       calcd_time_step=calcd_time_step,
        #                                       feat_exp=0.001,
        #                                       ovshoot_exp=0.05,
        #                                       time_exp=float(1),
        #                                       exp_error=0.05)
        print("roll fitness: {:.4f}".format(roll_fitness))
        

        #analyze pitch
        print("collecting pitch-fitness...")
        pitch_rot = np.array(bot_stateDF["pitch_rot"])
        x_exp = np.array(bot_stateDF["x_pos"])
        y_exp = np.array(bot_stateDF["y_pos"])
        #print("pitch-fitness feat: ({}, {})".format(float(fitness_feats[1][0]), float(fitness_feats[1][1])))
        pitch_trails, pitch_times = self.processStateFileFeats(pitch_rot, timestep=timestep)
        xpos_trials, xpos_times = self.processStateFileFeats(x_exp, timestep=timestep)
        ypos_trials, ypos_times = self.processStateFileFeats(y_exp, timestep=timestep)
        pitch_fitness = 0
        #pitch_fitness = self .getOvshootFitness(pitch_trails,
        #                                        pitch_times,
        #                                        calcd_time_step=calcd_time_step,
        #                                        feat_exp=0.02,
        #                                        ovshoot_exp=0.05,
        #                                        time_exp=float(1),
        #                                        exp_error=0.05)
        print("pitch fitness: {:.4f}".format(pitch_fitness))
        traj_fitness = 0
        #traj_fitness = self.getFitnessFromTrialsMSE(xpos_trials,
        #                                            xpos_times,
        #                                            ypos_trials, 
        #                                            ypos_times,
        #                                            feat_exp=(float(fitness_feats[1][0]),
        #                                                      float(fitness_feats[1][1])), 
        #                                            calcd_time_step=calcd_time_step)
        print("traj fitness: {:.4f}".format(traj_fitness))
        xpos_fitness = 0
        #xpos_fitness = self.getOvshootFitness(xpos_trials,
        #                                      xpos_times,
        #                                      calcd_time_step=calcd_time_step,
        #                                      feat_exp=float(fitness_feats[1][0]),
        #                                      ovshoot_exp=0.05,
        #                                      time_exp=float(3),
        #                                      exp_error=0.05)
        print("xpos fitness: {:.4f}".format(xpos_fitness))
        ypos_fitness = 0
        #ypos_fitness = self.getOvshootFitness(ypos_trials,
        #                                      ypos_times,
        #                                      calcd_time_step=calcd_time_step,
        #                                      feat_exp=float(fitness_feats[1][1]),
        #                                      ovshoot_exp=0.05,
        #                                      time_exp=float(3),
        #                                      exp_error=0.05)
        print("ypos fitness: {:.4f}".format(ypos_fitness))
        #analyze yaw
        print("collecting yaw-fitness...")
        yaw_rot = np.array(bot_stateDF["yaw_rot"])
        print("yaw-fitness feat: {}".format(fitness_feats[2]))
        yaw_trials, yaw_times = self.processStateFileFeats(yaw_rot, timestep=timestep)
        #yaw_fitness = 0
        yaw_fitness = self.getOvshootFitness(yaw_trials,
                                             yaw_times,
                                             calcd_time_step=calcd_time_step,
                                             feat_exp=fitness_feats[2],
                                             ovshoot_exp=0.05,
                                             time_exp=float(2),
                                             exp_error=0.05)
        print("yaw fitness: {:.4f}".format(yaw_fitness))
        

        
        #analyze thrust
        print("collecting thrust-fitness...")
        thrust = np.array(bot_stateDF["z_pos"])
        #print("thrust-fitness feat: {}".format(fitness_feats[3]))
        thrust_trials, thrust_times = self.processStateFileFeats(thrust, timestep=timestep)
        thrust_fitness = 0
        #thrust_fitness = self.getOvshootFitness(thrust_trials,
        #                                        thrust_times,
        #                                        calcd_time_step=calcd_time_step,
        #                                        feat_exp=float(fitness_feats[3]),
        #                                        ovshoot_exp=0.05,
        #                                        time_exp=float(3),
        #                                        exp_error=0.2)
        print("thrust fitness: {:.4f}".format(thrust_fitness))
        
        total_fitness = yaw_fitness
        print("total fitness: {:.4f}".format(total_fitness))
        
        #return total_fitness
        return total_fitness


    #update particle dictionary with params
    def updateParticleDict(self, params_edit, params):
        for param in params:
            print("curr-particle param: {:.4f}".format(param))
            
        #params_edit["x_Kp"] = params[0]
        #params_edit["x_Ki"] = params[1]
        #params_edit["x_Kd"] = params[2]
        #params_edit["y_Kp"] = params[3]
        #params_edit["y_Ki"] = params[4]
        #params_edit["y_Kd"] = params[5]
        #params_edit["pitch_Kp"] = params[6]
        #params_edit["pitch_Ki"] = params[7]
        #params_edit["pitch_Kd"] = params[8]
        #params_edit["roll_Kp"] = params[9]
        #params_edit["roll_Ki"] = params[10]
        #params_edit["roll_Kd"] = params[11]
        #params_edit["throttle_Kp"] = params[0]
        #params_edit["throttle_Ki"] = params[1]
        #params_edit["throttle_Kd"] = params[2]
        params_edit["yaw_Kp"] = params[0]
        params_edit["yaw_Ki"] = params[1]
        params_edit["yaw_Kd"] = params[2]
        
        return params_edit
    
    #update with final params for (roll, pitch, yaw, thrust)
    def updateParticleCSV(self, params, filedir, idx):
        optim_file = filedir + r"\optim_file_" + str(idx) + ".csv"
        clearFileIfExists(optim_file)

        with open(optim_file, 'a', newline='', encoding='utf-8') as opt_obj:
            w = csv.writer(opt_obj)
            w.writerows(params.items())
            opt_obj.close()


    #run version 2, callable for optimizer
    def run_v2(self, particle_params):
        self.iter_num += 1
        print("at iteration: {}----------------------".format(self.iter_num))
        
        #convert particle params np(20, 18) into dict-times

        #collect initial avg and std-dev for each param over all particles--(meas. convergence)
        #avgstd_arr = np.array([])
        param_avgstd = self.measAvgStdDevforParticles()
        #for key, val in param_avgstd.items():
        #    print("param: {}-->{}".format(key, val))
        self.avgstd_arr = np.append(self.avgstd_arr, param_avgstd)
        
        #array to hold performance for every particle
        perf_particle_arr = []
        #iterate over particles and collect fitness per particle
        for idx, params in enumerate(particle_params):
            print("-----------------particle: {}-----------------".format(idx + 1))

            t0 = self.getTime()

            calcd_time_step = self.time_step / 1000
            print("using time step: {}".format(calcd_time_step))
            
            #robot initial position and orientation
            bot_init_pos = [0, 0, 0.1]
            bot_rot_att = [0, -1, 0, 0.0701]
            #viewpoint initial position and orientation
            viewpt_init_pos = [6.05, -0.252, 2.5]
            viewpt_init_orient = [0.144, 0, -1, 3.17]

            #overwrite optim-file for robot to get params
            params_edit = self.particles[idx]
            update_param_dict = self.updateParticleDict(params_edit, params)
            #print("update params:\n{}".format(update_param_dict))
                
            ##update particle-csv file
            self.updateParticleCSV(update_param_dict, 
                                   self.files_dict['param-filedir'], 
                                   idx)

            #overwrite optim-file for robot to get params
            self.updateOptimFile(update_param_dict, self.files_dict)
            
            #Robot Node
            robot_node = self.getFromDef("mavic2pro")
            viewpoint_node = self.getFromDef("viewpoint")
            if robot_node is None:
                print("Robot Node not found in world file")
                sys.exit(1)
            else:
                trans_field = robot_node.getField("translation")
                rot_field = robot_node.getField("rotation")
                robot_node.restartController()
                print("Controller restarted...")
            
            #Viewpoint Node
            if viewpoint_node is None:
                print("Viewpoint doesn't exit in world")
                sys.exit(1)
            else:
                vwpt_pos_field = viewpoint_node.getField("position")
                vwpt_rot_field = viewpoint_node.getField("orientation")
                
            #collect overshoots for x, y, z, roll, pitch, yaw
            #clear state and pid files
            clearFileIfExists(self.files_dict['state-file'])
            clearFileIfExists(self.files_dict['pid-file'])
            
            #perform PSO with given overshoot expectations
            #PSO is done with each param one at a time
            #each param has its own overshoot expectation
            for i in range(3):
                print("count down: {}".format(self.getTime() - t0))
                while self.step(self.time_step) != -1:
                    if self.super_rcvr.getQueueLength() > 0:
                        #collecting messages from mavic2pro
                        #rcvd pos_msg
                        pos_msg = self.super_rcvr.getFloats()
                        #print("pos msg1: ({:.4f}, {:.4f}, {:.4f})".format(pos_msg[0], pos_msg[1], pos_msg[2]))
                        self.super_rcvr.nextPacket()
                        #rcvd attitude_msg
                        att_msg = self.super_rcvr.getFloats()
                        #print("att msg2: ({:.4f}, {:.4f}, {:.4f})".format(att_msg[0], att_msg[1], att_msg[2]))
                        self.super_rcvr.nextPacket()
                        #rcvd velocity_msg
                        vel_msg = self.super_rcvr.getFloats()
                        #print("vel msg3: ({:.4f}, {:.4f}, {:.4f})".format(vel_msg[0], vel_msg[1], vel_msg[2]))
                        self.super_rcvr.nextPacket()
                        #rcvd position-acceleration msg
                        pos_acc_msg = self.super_rcvr.getFloats()
                        #print("pos acc msg4: ({:.4f}, {:.4f}, {:.4f})".format(pos_acc_msg[0], pos_acc_msg[1], pos_acc_msg[2]))
                        self.super_rcvr.nextPacket()
                        #attitude-rotational-velocity msg
                        att_acc_msg = self.super_rcvr.getFloats()
                        #print("att acc msg5: {}".format(att_acc_msg))
                        self.super_rcvr.nextPacket()
                        #rcvd time_msg
                        time_msg = self.super_rcvr.getFloats()
                        #print("time msg6: {}".format(time_msg))
                        self.super_rcvr.nextPacket()
                        #get state of drone
                        bot_state_msg = struct.unpack('4s', self.super_rcvr.getBytes())[0].decode('utf-8')
                        #print("time msg7: {}".format(bot_state_msg))
                        self.super_rcvr.nextPacket()
                        front_mtr_msg = self.super_rcvr.getFloats()
                        #print("front-motors msg8: {}".format(front_mtr_msg))
                        self.super_rcvr.nextPacket()
                        rear_mtr_msg = self.super_rcvr.getFloats()
                        #print("rear-motors msg9: {}".format(rear_mtr_msg))
                        self.super_rcvr.nextPacket()


                        #update mavic_state csv
                        mavic_state = [pos_msg[0], pos_msg[1], pos_msg[2],#(x,y,z)
                                        att_msg[0], att_msg[1], att_msg[2],#(roll,pitch,yaw)
                                        vel_msg[0], vel_msg[1], vel_msg[2],#(dx/dt, dy/dt, dz/dt)
                                        pos_acc_msg[0], pos_acc_msg[1], pos_acc_msg[2],#(x_acc, y_acc, alt_acc)
                                        att_acc_msg[0], att_acc_msg[1], att_acc_msg[2],#(roll_acc, pitch_acc, yaw_acc)
                                        time_msg[0], #timestep
                                        front_mtr_msg[0], front_mtr_msg[1],#frnt_lft, frnt_rgt
                                        rear_mtr_msg[0], rear_mtr_msg[1]]#rear_lft, rear_rgt
                        self.writeMavicState(mavic_state, self.files_dict['state-file'])

                        posPID_msg = self.super_rcvr.getFloats()
                        #print("posPID msg9: {}".format(posPID_msg))
                        self.super_rcvr.nextPacket()
                        rpPID_msg = self.super_rcvr.getFloats()
                        #print("roll-pitchPID msg10: {}".format(rpPID_msg))
                        self.super_rcvr.nextPacket()
                        ytPID_msg = self.super_rcvr.getFloats()
                        #print("yaw-throttlePID msg11: {}".format(ytPID_msg))
                        self.super_rcvr.nextPacket()
                        rpInput_msg = self.super_rcvr.getFloats()
                        #print("roll-pitch inputs msg12: {}".format(rpInput_msg))
                        self.super_rcvr.nextPacket()
                        ytInput_msg = self.super_rcvr.getFloats()
                        #print("yaw-throttle inputs msg13: {}".format(ytInput_msg))
                        self.super_rcvr.nextPacket()
                        diff_alt_msg = self.super_rcvr.getFloats()
                        #print("diff-alt msg14: {}".format(diff_alt_msg))
                        self.super_rcvr.nextPacket()
                        que_len = self.super_rcvr.getQueueLength()
                        #print("que len: {}".format(que_len))
                        
                        #update PID_and_Inputs csv
                        pid_inputs = [posPID_msg[0], posPID_msg[1], #xposPID, yposPID
                                      rpPID_msg[0], rpPID_msg[1], #rollPID, pitchPID
                                      ytPID_msg[0], ytPID_msg[1], #yawPID, throttlePID
                                      rpInput_msg[0], rpInput_msg[1], #roll-input, pitch-input
                                      ytInput_msg[0], ytInput_msg[1], #yaw-input, throttle-input
                                      diff_alt_msg[0], diff_alt_msg[1]]#diff_alt, clampd_diff_alt
                        self.writePIDandInputs(pid_inputs, self.files_dict['pid-file'])

                        ##Pause simulation when conditions are met
                        #set exit-condition for time
                        time_bool = False
                        if self.getTime() > float(60):
                            time_bool = True
                        
                        #set exit-conditions for position
                        #--->2 conditions: if bot is in hover->stay at curr_waypt
                        #----------------->if bot is in pitch-chg state->swap pos cond with angle_tol
                        if bot_state_msg == 'grnd' or bot_state_msg == 'toff':
                            self.curr_waypnt = self.waypoints[0]
                            self.next_waypnt = self.waypoints[0]
                        elif bot_state_msg == 'hovr' and self.next_waypnt == self.curr_waypnt:
                            self.waypoint_idx += 1
                            self.next_waypnt = self.waypoints[self.waypoint_idx]
                            self.yaw_feat_exp = self.getYawChgAngle(self.curr_waypnt, self.next_waypnt, att_msg[2])
                        wypnt_tolerance = float(30)
                        position_cond = np.array([pos_msg[0], pos_msg[1], pos_msg[2]])
                        position_bool = False
                        #only checking for going above target altitude
                        if (position_cond[2] > self.curr_waypnt[2] + (wypnt_tolerance * 5)):#for altitude
                            position_bool = True
                            print("z became true")
                        
                        #if drone is in Pitch-Chg state-->get current pitch waypt-tgt(x, y)
                        if bot_state_msg == "pitd":
                            pitch_waypt = (pos_msg[0], pos_msg[1])
                        #else-->drone is not in Pitch-Chg state-->use current x-setpoint, y-setpoint
                        else:
                            pitch_waypt = (self.curr_waypnt[0], self.curr_waypnt[1])
                        #check x and y position cond
                        if position_cond[0] > pitch_waypt[0] + wypnt_tolerance or position_cond[0] < pitch_waypt[0] - wypnt_tolerance:
                            position_bool = True
                            print("x became true")
                        if position_cond[1] > pitch_waypt[1] + wypnt_tolerance or position_cond[1] < pitch_waypt[1] - wypnt_tolerance:
                            position_bool = True
                            print("y became true")
                            
                        #flag to see if drone made it to the last state/completed tuning_course
                        if bot_state_msg == 'land':
                            self.cleared_crs = True


                        #set exit-conditions for altitude velocity
                        alt_vel_bool = False
                        if vel_msg[2] < -float(5):
                            alt_vel_bool = True
                    
                        #set exit-conditions for attitude
                        attitude_cond_r_p = np.array([att_msg[0], att_msg[1]])#roll and pitch cond
                        attitude_cond_y = np.array([att_msg[2]])#yaw cond
                        attitude_bool = False
                        #only checking for roll and pitch since we change yaw
                        for att in attitude_cond_r_p:
                            if (att > 1.25) or (att < -1.25):
                                attitude_bool = True
                                print("pitch too much...")

                        #set exit-conditions for motor inputs
                        mtr_input_cond = np.array([front_mtr_msg[0], 
                                                   front_mtr_msg[1], 
                                                   rear_mtr_msg[0],
                                                   rear_mtr_msg[1]])
                        mtr_inpt_bool = False
                        for inpt in mtr_input_cond:
                            if inpt > 300:
                                mtr_inpt_bool = True
                                                            
                        #restart condition
                        reset_sim_conds = np.array([attitude_bool,
                                                    mtr_inpt_bool,
                                                    position_bool,
                                                    time_bool,
                                                    alt_vel_bool,
                                                    self.cleared_crs])

                        if attitude_bool or mtr_inpt_bool or position_bool or time_bool or alt_vel_bool:
                            #reset Sim
                            self.waypoint_idx = 0
                            self.curr_waypoint = [0, 0, 0]
                            self.next_waypoint = [0, 0, 0]
                            trans_field.setSFVec3f(bot_init_pos)
                            rot_field.setSFRotation(bot_rot_att)
                            vwpt_pos_field.setSFVec3f(viewpt_init_pos)
                            vwpt_rot_field.setSFRotation(viewpt_init_orient)
                            robot_node.resetPhysics()
                            robot_node.restartController()
                            self.simulationReset()
                            break
                    else:
                        print("Emitter-Queues are currently empty...")
            #perform PSO with collected results
            try:
                #expected-features for reference
                fitness_feats = [0, #used as available interchangable feature expectation
                                 self.next_waypnt, #next (x, y) tgt-waypt for roll/pitch
                                 self.yaw_feat_exp, #feature for yaw-tgt-heading
                                 update_param_dict['target_altitude']#feature for tgt-altitude
                                 ]
                print("using calcd yaw-feats...")
            except:
                #expected-features for reference
                fitness_feats = [0, #used as available interchangable feature expectation
                                 self.curr_waypnt, #next (x, y) tgt-waypt for roll/pitch
                                 float(0), #feature for yaw-tgt-heading
                                 update_param_dict['target_altitude']#feature for tgt-altitude
                                 ]
                print("not using calcd yaw-feats...")

            #collect particle's fitness performance 
            fitness = self.collectFitnessScore(self.files_dict,
                                               params,
                                               fitness_feats,
                                               calcd_time_step)
            perf_particle_arr.append(fitness)
            
        self.avgfit_arr = np.append(self.avgfit_arr, 
                                    np.mean(np.array(perf_particle_arr)))

        #write particle files (avg-std arr) (avg-fit arr)
        self.writeParticleFiles()

        for i, perf in enumerate(perf_particle_arr):
            print("particle {} has fitness: {:.4f}".format(i+1, perf))

        return np.array(perf_particle_arr)





#collect waypoints, from waypoint_logfile.txt
def collectWaypoints(param_filedir) -> list:
    waypoint_list = []
    filename = param_filedir + r"\waypoint_logfile.txt"
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            points = line.strip().split(" ")
            waypoint_list.append([float(points[0]), float(points[1]), float(points[2])])
    return waypoint_list


#return initial position params for Swarm
def getSwarmInitPositions(particles):
    partcl_swrm_dict = {}
    for idx, params in enumerate(particles):
        particle_data = np.array([#float(params["x_Kp"]),
                                  #float(params["x_Ki"]),
                                  #float(params["x_Kd"]),
                                  #float(params["y_Kp"]),
                                  #float(params["y_Ki"]),
                                  #float(params["y_Kd"])])
                                  #float(params["pitch_Kp"]),
                                  #float(params["pitch_Ki"]),
                                  #float(params["pitch_Kd"]),
                                  #float(params["roll_Kp"]),
                                  #float(params["roll_Ki"]),
                                  #float(params["roll_Kd"])])
                                  #float(params["throttle_Kp"]),
                                  #float(params["throttle_Ki"]),
                                  #float(params["throttle_Kd"])])
                                  float(params["yaw_Kp"]),
                                  float(params["yaw_Ki"]),
                                  float(params["yaw_Kd"])])
        #print("particle type: {}".format(type(particle_data[0])))
        partcl_swrm_dict[idx+1] = particle_data
        
    #for key, val in partcl_swrm_dict.items():
    #    print("particle{}:\n{}".format(key, val))
    particle_lst = list(partcl_swrm_dict.values())

    return np.array(particle_lst)


#return initial velocity for swarm
def getSwarmInitVelocities(num_particles):
    init_velocities = psb.generate_velocity(n_particles=num_particles,
                                            dimensions=3)
    #print("init vels:\n{}".format(init_velocities))
    return init_velocities


#given a filename, clear if it exists
def clearFileIfExists(filename):
    if os.path.isfile(filename):
        os.remove(filename)
        

#return bounds for PIDs
def getPIDBounds():
    #max terms
    p_term_max = 2.2
    i_term_max = 0.2
    d_term_max = 1.5

    #min terms
    p_term_min = 0.5
    i_term_min = 0.0008
    d_term_min = 0.1
    #max-min bounds arrays
    max_bounds = []
    min_bounds = []
    for i in range(1):
        #max-bounds for PID
        max_bounds.append(p_term_max)#P_max
        max_bounds.append(i_term_max)#I_max
        max_bounds.append(d_term_max)#D_max
        #min-bounds for PID
        min_bounds.append(p_term_min)#P_min
        min_bounds.append(i_term_min)#I_min
        min_bounds.append(d_term_min)#D_min

    for i in range(len(max_bounds)):
        print("bounds: ({}, {})".format(min_bounds[i], max_bounds[i]))
    
    return (np.array(min_bounds), np.array(max_bounds))



#create particles for PSO
def createAndCollectParticles(params_file, filedir, num_of_particles):
    particles = np.array([])
    for i in range(num_of_particles):
        init_dict = dict()
        optim_file = filedir + r"\optim_file_" + str(i) + ".csv"
        if os.path.isfile(optim_file):
            print("using optimizer file parameters...")
            with open(optim_file, "r") as f:
                lines = csv.reader(f)
                for line in lines:
                    init_dict[line[0]] = line[1]
                f.close()
        else:
            print("using initial random parameters")
            with open(params_file, "r") as f:
                lines = csv.reader(f)
                for line in lines:
                    init_dict[line[0]] = line[1]
                f.close()
            #PID values for xposPID
            #init_dict["x_Kp"] = np.random.uniform(0.9, 1.9)
            #init_dict["x_Ki"] = np.random.uniform(0.0009, 0.01)
            #init_dict["x_Kd"] = np.random.uniform(0.7, 1.0)
            #PID values for yposPID
            #init_dict["y_Kp"] = np.random.uniform(0.9, 1.9)
            #init_dict["y_Ki"] = np.random.uniform(0.0009, 0.01)
            #init_dict["y_Kd"] = np.random.uniform(0.7, 1.0)
            #PID values for pitchPID
            #init_dict["pitch_Kp"] = np.random.uniform(0.1, 1)
            #init_dict["pitch_Ki"] = np.random.uniform(0.01, 0.2)
            #init_dict["pitch_Kd"] = np.random.uniform(0.1, 1)
            #PID values for rollPID
            #init_dict["roll_Kp"] = np.random.uniform(0.1, 1)
            #init_dict["roll_Ki"] = np.random.uniform(0.01, 0.2)
            #init_dict["roll_Kd"] = np.random.uniform(0.1, 1)
            #init_dict["throttle_Kp"] = np.random.uniform(1.7, 2.0)
            #init_dict["throttle_Ki"] = np.random.uniform(0.1, 0.4)
            #init_dict["throttle_Kd"] = np.random.uniform(0.5, 1.0)
            init_dict["yaw_Kp"] = np.random.uniform(0.6, 2)
            init_dict["yaw_Ki"] = np.random.uniform(0.01, 0.1)
            init_dict["yaw_Kd"] = np.random.uniform(0.3, 1.2)
            #delete params file if it is not open
            #and collected dictionary for parameters
            if f.closed == True:
                clearFileIfExists(params_file)
            #rewrite params file with initialized random variables
            with open(params_file, 'a', newline='', encoding='utf-8') as init_obj:
                w = csv.writer(init_obj)
                w.writerows(init_dict.items())
                init_obj.close()
            with open(optim_file, 'a', newline='', encoding='utf-8') as opt_obj:
                w = csv.writer(opt_obj)
                w.writerows(init_dict.items())
                opt_obj.close()
        particles = np.append(particles, init_dict)

    return particles


#main method
def main():
    print("numpy version: {}".format(np.__version__))
    param_filedir = r"C:\Users\ericx\OneDrive\Desktop\mine\grad_school\MARS_Lab_Swarm\Gazebo_dragonFly\webots_tutorial\hovering_test_project\controllers\PID_params_exp_PSO"
    params_file = param_filedir + "\default_params_edit.csv"
    optim_file = param_filedir + "\optim_edit.csv"
    num_of_particles = 20
    #collect list of initial particles(each particle is a dictionary)
    particles = createAndCollectParticles(params_file=params_file, 
                                  filedir=param_filedir, 
                                  num_of_particles=num_of_particles)
    
    #print(particles)
    print("Parameters Written")
    #clear output files if they exit
    state_filedir = r"C:\Users\ericx\OneDrive\Desktop\mine\grad_school\MARS_Lab_Swarm\Gazebo_dragonFly\webots_tutorial\hovering_test_project\python_utilities"
    ###experimental-designator filename
    state_filename = state_filedir + r"\mavic_state7.csv"
    pid_filename = state_filedir + r"\PID_and_inputs7.csv"
    clearFileIfExists(state_filename)
    clearFileIfExists(pid_filename)

    #used for saving pbest parameters per iteration
    pbest_filename = param_filedir + r"\pbest_params.csv"
    clearFileIfExists(pbest_filename)
    
    #used for saving gbest parameters per iteration
    gbest_filename = state_filedir + r"\gbest_params.npy"
    clearFileIfExists(gbest_filename)
    files_dict = {'state-filedir': state_filedir,
                  'param-filedir': param_filedir,
                  'state-file': state_filename, 
                  'pid-file': pid_filename,
                  'optim-file': optim_file, 
                  'gbest-file': gbest_filename,
                  'pbest-file': pbest_filename}

    #perform PSO
    #get timestep and initialize robot
    init_params = particles[0]
    TIME_STEP = int(init_params["QUADCOPTER_TIME_STEP"])
    robot = CrazyFlieSuper(TIME_STEP)
    robot.files_dict = files_dict
    #collect waypoints for PSO
    robot.waypoints = collectWaypoints(param_filedir)
    robot.particles = particles

    #collect bounds for PSO-optimizer
    pid_bounds = getPIDBounds()
    print("no. of bounds: {}".format(len(pid_bounds)))
    for bound in pid_bounds:
        print(bound)
    print("Bounds created...")
   
    #create PSO-Velocity terms
    wgt = np.full((1, ), 0.9) #implement weight update later, if needed
    #c1 = np.random.random_sample((1,))
    c1 = np.full((1, ), 0.5)
    #c2 = np.random.random_sample((1,))
    c2 = np.full((1, ), 0.2)
    print("wgt: {}".format(wgt))
    print("c1: {}".format(c1))
    print("c2: {}".format(c2))
    options = {'c1': c1, 'c2': c2, 'w': wgt}

    #get Swarm initial position
    swrm_init_positions = getSwarmInitPositions(particles)
    #get Swarm initial velocity
    swrm_init_velocities = getSwarmInitVelocities(num_of_particles)
    #create Swarm for optimizer
    particle_swarm = Swarm(position=swrm_init_positions,
                           velocity=swrm_init_velocities,
                           options=options)
    #print(particle_swarm)
    print("Swarm created...")

    #collect waypoints
    #waypoints = collectWaypoints(param_filedir)
    #print(waypoints)

    #create optimizer-->poss. no need to create initial swarm
    pso_optimizer = ps.single.GlobalBestPSO(n_particles=num_of_particles,
                                            dimensions=3,
                                            options=options,
                                            bounds=pid_bounds,
                                            velocity_clamp=(0.8, 0.0001),
                                            init_pos=swrm_init_positions)
    #run optimizer and collect stats
    #pso_optmizr_stats = pso_optimizer.optimize(robot.run(particles, 
    #                                                     avgstd_arr, 
    #                                                     files_dict, 
    #                                                     waypoints),
    #                                           iters=2)

    #collect optimizer stats (gbest, bestfitness)
    pso_optmizr_stats = pso_optimizer.optimize(robot.run_v2,
                                               iters=70)

    #do final write of particles
    robot.writeParticleFiles()
    
    #save gbest params from optimizer
    gbest_file = state_filedir + r"\gbest_params.npy"
    gbest_params = np.array(pso_optmizr_stats[1])
    clearFileIfExists(gbest_file)
    with open(gbest_file, 'wb') as gb_file:
        np.save(gbest_file, gbest_params)
        gb_file.close()
    
    #pause sim
    robot.simulationSetMode(0)


if __name__ == '__main__':
    main()