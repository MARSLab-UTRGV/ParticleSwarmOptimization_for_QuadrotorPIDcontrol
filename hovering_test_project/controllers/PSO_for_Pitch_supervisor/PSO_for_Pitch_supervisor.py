"""PSO for Pitch-Supervisor Controller
1. Supervisor for PSO with PID tuning for Pitch tuning
2. Params Domain: 
    {'P': [0, 1]}
    {'I': [0, 1]}
    {'D': [0, 1]}
3. PSO:
    pbest_obj-fcn = use MSE over reference traj and exp. traj
    X = {PID parameters} --(maybe try to input tgt-altitude)
    V = initial set to same-dims as X, all 0s
    V_update = wgt*V + c1*r1*(pbest -X) + c2*r2*(gbest -X)
4. Reset hover_pitch_exp_PSO.py Controller based on failing conditions
    a. find fitness score for failed conditions
"""


# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
###IMPORTS#######################################
from re import X
import sys, os
import csv, math
import struct
from controller import Robot, Supervisor
from cmath import pi

try: 
    import numpy as np
    import pandas as pd
    from csv import DictWriter
    from scipy.signal import find_peaks
    from simple_pid import PID
    from sklearn.metrics import mean_squared_error
except ImportError:
    sys.exit("Warning: 'numpy' module not found.")
#################################################


class CrazyFlieSuper(Supervisor):
    def __init__(self, time_step):
        Supervisor.__init__(self)

        try:
            self.time_step = time_step
        except:
            self.time_step = int(self.getBasicTimeStep())
        print("Supervisor using timestep: {}".format(self.time_step))

        # Get and enable devices
        #include emitter and receiver to get comms with Supervisor
        self.super_rcvr = self.getDevice("super_rcvr")
        self.super_rcvr.enable(self.time_step)
        self.super_emtr = self.getDevice("super_emtr")

        self.waypoint_idx = 0
        self.curr_waypnt = [0, 0, 0]
        self.next_waypnt = [0, 0, 0]
        
    
    def run(self, params, files_dict, waypoints):
        t0 = self.getTime()

        #print("super calcd time step: {}".format((self.time_step * 4) / 1000))
        #calcd_time_step = (self.time_step * 4) / 1000
        #print("super calcd time step: {}".format((self.time_step * 8) / 1000))
        #calcd_time_step = (self.time_step * 8) / 1000
        calcd_time_step = self.time_step / 1000
        print("using time step: {}".format(calcd_time_step))
        
        #robot initial position and orientation
        bot_init_pos = [0, 0, 0.1]
        bot_rot_att = [0, -1, 0, 0.0701]
        #viewpoint initial position and orientation
        viewpt_init_pos = [6.05, -0.252, 2.5]
        viewpt_init_orient = [0.144, 0, -1, 3.17]
        

        #overwrite optim-file for robot to get params
        updateOptimFile(params, files_dict)
        
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
        clearFileIfExists(files_dict['state-file'])
        clearFileIfExists(files_dict['pid-file'])
        
        #perform PSO with given overshoot expectations
        #PSO is done with each param one at a time
        #each param has its own overshoot expectation
        for i in range(4):
            print("count down: {}".format(self.getTime() - t0))
            while self.step(self.time_step) != -1:
                if self.super_rcvr.getQueueLength() > 0:
                    #collecting messages from mavic2pro
                    pos_msg = self.super_rcvr.getFloats()
                    #print("pos msg1: ({:.4f}, {:.4f}, {:.4f})".format(pos_msg[0], pos_msg[1], pos_msg[2]))
                    self.super_rcvr.nextPacket()
                    att_msg = self.super_rcvr.getFloats()
                    #print("att msg2: ({:.4f}, {:.4f}, {:.4f})".format(att_msg[0], att_msg[1], att_msg[2]))
                    self.super_rcvr.nextPacket()
                    vel_msg = self.super_rcvr.getFloats()
                    #print("vel msg3: ({:.4f}, {:.4f}, {:.4f})".format(vel_msg[0], vel_msg[1], vel_msg[2]))
                    self.super_rcvr.nextPacket()
                    pos_acc_msg = self.super_rcvr.getFloats()
                    #print("pos acc msg4: ({:.4f}, {:.4f}, {:.4f})".format(pos_acc_msg[0], pos_acc_msg[1], pos_acc_msg[2]))
                    self.super_rcvr.nextPacket()
                    att_acc_msg = self.super_rcvr.getFloats()
                    #print("att acc msg5: {}".format(att_acc_msg))
                    self.super_rcvr.nextPacket()
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
                    writeMavicState(mavic_state, files_dict['state-file'])
                    
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
                    writePIDandInputs(pid_inputs, files_dict['pid-file'])
                    
                    ##Pause simulation when conditions are met
                    #set exit-condition for time
                    time_bool = False
                    if self.getTime() > float(60):
                        time_bool = True

                    #set exit-conditions for position
                    #--->2 conditions: if bot is in hover->stay at curr_waypt
                    #----------------->if bot is in pitch-chg state->swap pos cond with angle_tol
                    if bot_state_msg == 'grnd' or bot_state_msg == 'toff':
                        self.curr_waypnt = waypoints[0]
                        self.next_waypnt = waypoints[0]
                    elif bot_state_msg == 'hovr' and self.next_waypnt == self.curr_waypnt:
                        self.waypoint_idx += 1
                        self.next_waypnt = waypoints[self.waypoint_idx]
                        yaw_feat_exp = getYawChgAngle(self.curr_waypnt, self.next_waypnt, att_msg[2])
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
            fitness_feats = [0, self.next_waypnt, yaw_feat_exp, params['target_altitude']]
            print("using calcd yaw-feats...")
        except:
            fitness_feats = [0, self.curr_waypnt, float(0), params['target_altitude']]
            print("not using calcd yaw-feats...")
            
        fitness = collectFitnessScore(files_dict, params, fitness_feats, calcd_time_step)
        
        #self.simulationQuit(1)
        return fitness


#display error between flight trajectory and 
#get true-trajectory distance from center
def getFlightMSE(x_true, y_true, x_traj, y_traj):
    true_dist = np.sqrt(np.square(x_true) + np.square(y_true))
    true_dist_avg = np.round(np.mean(true_dist), 4)
    print("true_dist_avg: {}".format(true_dist_avg))
    true_dist = np.full(len(x_traj), true_dist_avg)
    #print(true_dist)
    #print(len(true_dist))
    #get flight-trajectory distance from cetner
    flgt_dist = np.sqrt(np.square(x_traj) + np.square(y_traj))
    print("flgt_dist avg: {}".format(np.mean(flgt_dist)))
    print(len(flgt_dist))
    #collect mse(true vs. flight)
    if np.mean(flgt_dist) < float(1):
        flgt_MSE = float(99)
    else:
        flgt_MSE = mean_squared_error(true_dist, flgt_dist)
    print("MSE: {:.4f}".format(flgt_MSE))
    return flgt_MSE


#collect exp-traj with ref-traj during pitch-tuning
#meas. MSE(exp-traj, ref-traj)
def getFitnessFromTrialsMSE(x_trials, 
                             x_trials_times, 
                             y_trials,
                             y_trials_times,
                             feat_exp=(float(0), float(0)),
                             calcd_time_step=0.032):
    fitness_avg_arr = []
    print("# of trials: {}".format(len(x_trials)))
    for i in range(len(x_trials)):
        x_trial = x_trials[i]
        x_trial_time = x_trials_times[i]
        y_trial = y_trials[i]
        y_trial_time = y_trials_times[i]
        print("len of x_trial: {}".format(len(x_trial)))
        print("len of y_trial: {}".format(len(y_trial)))
        
        if len(x_trial) < 1000:
            fitness_avg_arr.append(float(99))
            print("MSE: {:.4f}".format(float(99)))
        else:
            init_waypt = (x_trial[0], y_trial[0])
            step = y_trial_time[1] - y_trial_time[0]
            pitch_refpath = np.array(generatePitchRef(init_waypt, feat_exp, step, num_of_waypoints=len(x_trial)))
        
            x_ref = np.array([])
            y_ref = np.array([])
            for i, ref_pt in enumerate(pitch_refpath):
                x_ref = np.append(x_ref, ref_pt[0])
                y_ref = np.append(y_ref, ref_pt[1])
        
            trajMSE = getFlightMSE(x_ref, y_ref, x_trial, y_trial)
            fitness_avg_arr.append(trajMSE)
        
    return np.mean(np.array([fitness_avg_arr]))


#overshoot is collect when,
#abs(sum of peaks until it falls below over shoot - expected ovshoot)
def getFitnessFromTrialsITAE(trials, 
                             trials_times, 
                             calcd_time_step=0.032, 
                             feat_exp=0.2, 
                             time_exp=3.0):
    fitness_avg_arr = []
    print("# of trials: {}".format(len(trials)))
    for i, trial in enumerate(trials):
        exp_error = 0.001#add as parameter for various PID outputs
        trial_time = trials_times[i]
        print("len of trial: {}".format(len(trial)))
        peaks, peak_props = find_peaks(trial, height=feat_exp, distance=10)
        print("No. of peaks found: {}".format(len(peaks)))

        peak_bal_val = 0
        peak_cnt = 0
        peak_sum = 0
        while peak_bal_val == 0 and peak_cnt < len(peaks):
            peak = peaks[peak_cnt]
            peak_val = trial[peak]
            peak_sum += peak_val
            #print("peak-{}: {:.4f}".format(peak_cnt, peak_val))
            peak_err = abs(peak_val - feat_exp)
            #print("peak-error: {:.4f}".format(peak_err))
            if peak_err < exp_error:
                peak_sum += peak_val
                peak_bal_val = peak_val
            peak_cnt += 1
        #print("final peak val: {:.4f}".format(peak_bal_val))
        time_indices = []
        for i, val in enumerate(trial):
            if val == peak_bal_val:
                time_indices.append(i)

        if peak_cnt == 0:
            peak_avg = 500
        else:
            peak_avg = peak_sum / peak_cnt
        #print("peak_sum: {:.4f}".format(peak_sum))
        #print("peak_cnt: {}".format(peak_cnt))
        #print("peak_avg: {:.4f}".format(peak_avg))
        overshoot = abs(peak_avg - feat_exp)
        #print("overshoot: {:.4f}".format(overshoot))

        ovshot_exp = 0.2#add as parameter
        adjst_time_exp = time_exp
        #print("init. trial time: {}".format(calcd_time_step))
        if len(time_indices) > 0 and peak_bal_val != 0:
            adjst_time = calcd_time_step * time_indices[0]
        else:
            adjst_time = 9999

        print("adjust-time: {:.4f}".format(adjst_time))
        fitness = np.log10((adjst_time/adjst_time_exp) + 1) + np.log10((overshoot/ovshot_exp) + 1)
        print("fitness: {:.4f}".format(fitness))
        fitness_avg_arr.append(fitness)

    print("average fitness: {}".format(np.mean(np.array(fitness_avg_arr))))
    return np.mean(np.array(fitness_avg_arr))


#find way to return a single list of times from each feature
def processStateFileFeats(input_arr, timestep):
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


#collect fitness score from state-records
def collectFitnessScore(files_dict, params, fitness_feats, calcd_time_step=0.032):
    bot_state_dir = files_dict["state-file"]
    #print("bot-state-dir: {}".format(bot_state_dir))
    print("collecting fitness for particles...")
    cols = ['x_pos', 'y_pos', 'z_pos', 
            'roll_rot', 'pitch_rot', 'yaw_rot', 
            'x_vel', 'y_vel', 'alt_vel',
            'x_acc', 'y_acc', 'alt_acc',
            'roll_acc', 'pitch_acc', 'yaw_acc',
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
    x_exp = np.array(bot_stateDF["x_pos"])
    y_exp = np.array(bot_stateDF["y_pos"])
    print("roll-fitness feat: ({}, {})".format(float(fitness_feats[1][0]), float(fitness_feats[1][1])))
    roll_trials, roll_times = processStateFileFeats(roll_rot, timestep=timestep)
    xpos_trials, xpos_times = processStateFileFeats(x_exp, timestep=timestep)
    ypos_trials, ypos_times = processStateFileFeats(y_exp, timestep=timestep)
    #roll_fitness = getFitnessFromTrialsMSE(xpos_trials,
    #                                        xpos_times,
    #                                        ypos_trials, 
    #                                        ypos_times,
    #                                        feat_exp=(float(fitness_feats[1][0]),
    #                                                 float(fitness_feats[1][1])), 
    #                                        calcd_time_step=calcd_time_step)

    #analyze pitch
    print("collecting pitch-fitness...")
    pitch_rot = np.array(bot_stateDF["pitch_rot"])
    pitch_trails, pitch_times = processStateFileFeats(pitch_rot, timestep=timestep)
    #print("pitch-fitness feat: ({}, {})".format(float(fitness_feats[1][0]), float(fitness_feats[1][1])))
    pitch_fitness = getFitnessFromTrialsMSE(xpos_trials,
                                            xpos_times,
                                            ypos_trials, 
                                            ypos_times,
                                            feat_exp=(float(fitness_feats[1][0]),
                                                     float(fitness_feats[1][1])), 
                                            calcd_time_step=calcd_time_step)

    #analyze yaw
    print("collecting yaw-fitness...")
    yaw_rot = np.array(bot_stateDF["yaw_rot"])
    yaw_trials, yaw_times = processStateFileFeats(yaw_rot, timestep=timestep)
    #yaw_fitness = getFitnessFromTrialsITAE(yaw_trials,
    #                                  yaw_times,
    #                                   calcd_time_step=calcd_time_step,
    #                                   feat_exp=fitness_feats[2],
    #                                   time_exp=float(12))

    #analyze thrust
    print("collecting thrust-fitness...")
    thrust = np.array(bot_stateDF["z_pos"])
    thrust_trials, thrust_times = processStateFileFeats(thrust, timestep=timestep)
    #thrust_fitness = getFitnessFromTrialsITAE(thrust_trials,
    #                                      thrust_times,
    #                                      calcd_time_step=calcd_time_step,
    #                                      feat_exp=fitness_feats[3],
    #                                      time_exp=float(12))

    #total_fitness = roll_fitness + pitch_fitness + yaw_fitness + thrust_fitness
    total_fitness = pitch_fitness

    return total_fitness


def generatePitchRef(init_waypt, tgt_waypt, calcd_timestep, num_of_waypoints=0):
    calcd_dist = np.sqrt(((tgt_waypt[1] - init_waypt[1]) ** 2) + ((tgt_waypt[0] - init_waypt[0]) ** 2))
    print("creating {} waypoints for pitch-MSE".format(num_of_waypoints))

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


#return reference yaw angle of yaw-tuning
def getYawChgAngle(curr_waypt, tgt_waypt, curr_yaw):
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


#write PID and DOF-inputs for mavic2pro/robot-node
def writePIDandInputs(input_arr, filename):
    field_names = ['xposPD', 'yposPD',
                   'rollPID', 'pitchPID', 
                   'yawPID', 'throttlePID',
                   'roll_input', 'pitch_input', 
                   'yaw_input', 'vertical_input',
                   'diff_altitude', 'clampd_diff_altitude']

    csv_dict = {'xposPD': input_arr[0],
                'yposPD': input_arr[1],
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
        dictWriter_obj.writerow(csv_dict)
        f_obj.close()


#writes state for mavic2pro/robot-node
def writeMavicState(input_arr, filename):
    field_names = ['x_pos', 'y_pos', 'z_pos', 
                   'roll_rot', 'pitch_rot', 'yaw_rot', 
                   'x_vel', 'y_vel', 'alt_vel',
                   'x_acc', 'y_acc', 'alt_acc',
                   'roll_acc', 'pitch_acc', 'yaw_acc',
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
                'roll_acc': input_arr[12],
                'pitch_acc': input_arr[13],
                'yaw_acc': input_arr[14],
                'timestep': input_arr[15],
                'front_left_motor': input_arr[16],
                'front_right_motor': input_arr[17],
                'rear_left_motor': input_arr[18],
                'rear_right_motor': input_arr[19]}

    with open(filename, 'a', newline='', encoding='utf-8') as f_obj:
        dictWriter_obj = DictWriter(f_obj, fieldnames=field_names)
        dictWriter_obj.writerow(csv_dict)
        f_obj.close()


#update particle's best fitness
def checkPBestFitness(params, param_fitness, pbest_dict, dict_idx):
    print("p{} has fitness: {}".format(dict_idx, pbest_dict[dict_idx][1]))
    pbest_params = pbest_dict[dict_idx]
    print("pbest-params to verify: {}".format(pbest_params[0]))
    if param_fitness < pbest_params[1]:
        pbest_dict[dict_idx] = (params['y_Kp'], param_fitness)
        print("p{} now has fitness: {}".format(dict_idx, pbest_dict[dict_idx][1]))
        print("p{} changed to: {}".format(dict_idx, pbest_dict[dict_idx][0]))
    else:
        print("p{} remains at: {} with fitness: {}".format(dict_idx, pbest_dict[dict_idx][0], pbest_dict[dict_idx][1]))
    return pbest_dict


#update particle with best fit over all group
def checkGBestFitness(params, param_fitness, gbest):
    print("current gbest fitness: {}".format(gbest[1]))
    if param_fitness < gbest[1]:
        gbest = (params, param_fitness)
    print("returned gbest-fit: {}".format(gbest[1]))
    return gbest


#update optim-file with new waypoint
def updateOptimFile(params, files_dict):
    optim_file = files_dict["optim-file"]
    clearFileIfExists(optim_file)
    with open(optim_file, 'a', newline='', encoding='utf-8') as opt_obj:
        w = csv.writer(opt_obj)
        w.writerows(params.items())
        opt_obj.close()
    print("optim file updated...")


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


#measure the avg. and std. dev of distribution for params for particles
def measAvgStdDevforParticles(particles):
    print("collecting mean and std-dev for params")
    particles_list = []
    parent_file = os.path.dirname(os.getcwd())
    optim_dir = os.path.normpath(os.path.join(parent_file + "\hover_pitch_exp_PSO"))
    for path, curr_dir, files in os.walk(optim_dir):
        for file in files:
            if file.startswith("optim_file"):
                particles_list.append(pd.read_csv(optim_dir + "\\" + file, header=None))

    particleDF = pd.DataFrame(columns=particles_list[0][0])
    for i, particle in enumerate(particles_list):
        print("particle no. {}".format(i))
        particle_dict = pd.pivot_table(particle, values=1, columns=[0]).to_dict('records')
        particleDF = pd.concat([particleDF, pd.DataFrame.from_records(particle_dict)])
    
    params_obsvd = ["x_Kp", "x_Kd",
                    "y_Kp", "y_Kd",
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


#init. PBest for Particles
def createPBestParamsFit(particles):
    pbest_dict = {}
    init_fitness = 999
    for i, particle_dict in enumerate(particles):
        pbest_dict[i + 1] = (particle_dict['y_Kp'], init_fitness)
    return pbest_dict


#given a filename, clear if it exists
def clearFileIfExists(filename):
    if os.path.isfile(filename):
        os.remove(filename)


#update with final params for (roll, pitch, yaw, thrust)
def updateParticleCSV(new_params, params, filedir, idx):
    #update parameters
    #params["x_Kp"] = new_params[0]
    #params["x_Kd"] = new_params[0]
    params["y_Kp"] = new_params[0]
    #params["y_Kd"] = new_params[0]

    #params["throttle_Kp"] = new_params[0]
    #params["throttle_Ki"] = new_params[0]
    #params["throttle_Kd"] = new_params[0]
    #params["roll_Kp"] = new_params[0]
    #params["roll_Ki"] = new_params[1]
    #params["roll_Kd"] = new_params[0]
    #params["pitch_Kp"] = new_params[0]
    #params["pitch_Ki"] = new_params[4]
    #params["pitch_Kd"] = new_params[0]
    #params["throttle_Kp"] = new_params[0]
    #params["throttle_Ki"] = new_params[0]
    #params["throttle_Kd"] = new_params[0]
    #params["yaw_Kp"] = new_params[0]
    #params["yaw_Ki"] = new_params[0]
    #params["yaw_Kd"] = new_params[0]

    optim_file = filedir + r"\optim_file_" + str(idx) + ".csv"
    clearFileIfExists(optim_file)

    with open(optim_file, 'a', newline='', encoding='utf-8') as opt_obj:
        w = csv.writer(opt_obj)
        w.writerows(params.items())
        opt_obj.close()


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
            #init_dict["x_Kp"] = np.random.uniform(0, 2)
            #init_dict["x_Kd"] = np.random.uniform(0, 2)
            init_dict["y_Kp"] = np.random.uniform(0, 2)
            #init_dict["y_Kd"] = np.random.uniform(0, 2)
            #init_dict["pitch_Kp"] = np.random.uniform(0, 2)
            #init_dict["pitch_Ki"] = np.random.uniform(0, 2)
            #init_dict["pitch_Kd"] = np.random.uniform(0, 1)
            #init_dict["roll_Kp"] = np.random.uniform(0, 1)
            #init_dict["roll_Ki"] = np.random.uniform(0, 2)
            #init_dict["roll_Kd"] = np.random.uniform(0, 2)
            #init_dict["throttle_Kp"] = np.random.uniform(0, 2)
            #init_dict["throttle_Ki"] = np.random.uniform(0.0001, 0.1)
            #init_dict["throttle_Kd"] = np.random.uniform(0, 2)
            #init_dict["yaw_Kp"] = np.random.uniform(0, 2)
            #init_dict["yaw_Ki"] = np.random.uniform(0, 0.01)
            #init_dict["yaw_Kd"] = np.random.uniform(0, 1)
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


#PSO-supervisor main method
def main():
    print("numpy version: {}".format(np.__version__))
    param_filedir = r"C:\Users\ericx\OneDrive\Desktop\mine\grad_school\MARS_Lab_Swarm\Gazebo_dragonFly\webots_tutorial\hovering_test_project\controllers\hover_pitch_exp_PSO"
    params_file = param_filedir + "\params_edit.csv"
    optim_file = param_filedir + "\optim_edit.csv"
    num_of_particles = 20
    #collect list of initial particles(each particle is a dictionary)
    particles = createAndCollectParticles(params_file=params_file, 
                                  filedir=param_filedir, 
                                  num_of_particles=num_of_particles)
    print("Parameters Written")
    #clear output files if they exit
    state_filedir = r"C:\Users\ericx\OneDrive\Desktop\mine\grad_school\MARS_Lab_Swarm\Gazebo_dragonFly\webots_tutorial\hovering_test_project\python_utilities"
    ###experimental-designator filename
    state_filename = state_filedir + r"\mavic_state4.csv"
    pid_filename = state_filedir + r"\PID_and_inputs4.csv"
    clearFileIfExists(state_filename)
    clearFileIfExists(pid_filename)
    
    #used for saving pbest parameters per iteration
    pbest_filename = param_filedir + r"\pbest_params.csv"
    clearFileIfExists(pbest_filename)
    
    #used for saving gbest parameters per iteration
    gbest_filename = state_filedir + r"\gbest_params.npy"
    clearFileIfExists(gbest_filename)
    files_dict = {'state-file': state_filename, 
                  'pid-file': pid_filename,
                  'optim-file': optim_file, 
                  'gbest-file': gbest_filename,
                  'pbest-file': pbest_filename}
    
    #perform PSO
    #get timestep and initialize robot
    init_params = particles[0]
    TIME_STEP = int(init_params["QUADCOPTER_TIME_STEP"])
    robot = CrazyFlieSuper(TIME_STEP)
    
    #collect initial pbest for each particle and gbest
    pbest_dict = createPBestParamsFit(particles=particles)
    for key, val in pbest_dict.items():
        print("particle-{}--> {:.4f}".format(key, float(val[0])))
    gbest = pbest_dict[1]
    print("init-Gbest: {}".format(gbest))
    
    #V-particle velocity
    V_parts = np.zeros(1)
    print("initialized V_parts: {}".format(V_parts))
    
    #collect initial avg and std-dev for each param over all particles--(meas. convergence)
    avgstd_arr = np.array([])
    param_avgstd = measAvgStdDevforParticles(particles)
    avgstd_arr = np.append(avgstd_arr, param_avgstd)
    
    #collect waypoints
    waypoints = collectWaypoints(param_filedir)
    print(waypoints)
    
    #for 100 iterations, do 15 trials to find avg. overshoot for PSO
    iter_num = 30
    fitness_arr = np.array([])
    for i in range(iter_num):

        print("-----------------iteration: {}-----------------".format(i))
        #print particle params for iteration
        print("params for iter-{}".format(i))
        for j, param in enumerate(particles):
            print("p{}: {}".format(j, param['y_Kp']))

        #print particle's best params, so far
        print("pbests for iter-{}:".format(i))
        for key, val in pbest_dict.items():
            print("p{}: ({}: {})".format(key, val[0], val[1]))
            
        avg_fit_arr = np.array([])
        for idx, params in enumerate(particles):
            print("-----------------particle: {}-----------------".format(idx + 1))
            print("running particle-{} with params: {}".format(idx+1, params['y_Kp']))
            print("pbest: {}---{}".format(pbest_dict[idx+1][0], pbest_dict[idx+1][1]))
            
            #collect fitness from given particle parameters --> X = params
            param_fitness = robot.run(params, files_dict, waypoints)
            print("total-fitness: {}".format(param_fitness))
            avg_fit_arr = np.append(avg_fit_arr, param_fitness)
            
            #get pbest params
            print("current pbest: ({}: {})".format(pbest_dict[idx+1][0], pbest_dict[idx+1][1]))
            pbest_dict = checkPBestFitness(params=params, 
                                          param_fitness=param_fitness,
                                          pbest_dict=pbest_dict,
                                          dict_idx=idx + 1)
            pbest_tup = pbest_dict[idx + 1]

            #print("pbest:\n{}".format(pbest_tup[0]))
            #print("gbest:\n{}".format(gbest_tup[0]))

            #get gbest params
            gbest = checkGBestFitness(params=params,
                                      param_fitness=param_fitness,
                                      gbest=gbest)
            
            #create PSO-Velocity terms
            wgt = np.full((1, ), 0.09) #implement weight update later, if needed
            c1 = np.full((1, ), 0.1)
            c2 = np.full((1, ), 0.1)
            r1 = np.random.random_sample((1,))
            r2 = np.random.random_sample((1,))
            print("wgt: {}".format(wgt))
            print("c1: {}".format(c1))
            print("c2: {}".format(c2))
            print("r1: {}".format(r1))
            print("r2: {}".format(r2))

            #collect velocity for particles
            pbest_params = pbest_tup[0]
            gbest_params = gbest[0]
            #print("using pbest:\n{}".format(pbest_params))
            #print("using gbest:\n{}".format(gbest_params))

            pbest_arr = np.array([float(pbest_params)])

            gbest_arr = np.array([float(gbest_params["y_Kp"])])

            params_arr = np.array([float(params["y_Kp"])])
    
            print("pbest:\n{}".format(pbest_arr))
            print("gbest:\n{}".format(gbest_arr))
            print("params:\n{}".format(params_arr))

            first_term = np.multiply(wgt, V_parts)
            second_term_a = c1*r1
            second_term_b =  pbest_arr - params_arr
            second_term = second_term_a * second_term_b
            third_term_a = c2*r2
            third_term_b = gbest_arr - params_arr
            third_term = third_term_a * third_term_b
            
            #Velocity for particles
            V_parts = np.add(np.add(first_term, second_term), third_term)
            print("V_parts: {}".format(V_parts))
            new_params = params_arr + V_parts
            print("new params: {}".format(new_params))

            #Verify parameters are not above/below limits for respective PD/PID
    
            #write params to optim_file
            updateParticleCSV(new_params, params, param_filedir, idx)
            print("updated particle: {} at iter-{}".format(idx+1, i))
            
            #save gbest for iteration
            with open(files_dict["gbest-file"], 'wb') as gbest_file:
                np.save(gbest_file, gbest_arr)
                gbest_file.close()
                
        avg_fitness = avg_fit_arr.mean()
        std_fitness = avg_fit_arr.std()
        print("fit-avg over particles: {:.4f}".format(avg_fitness))
        print("fit-std over particles: {:.4f}".format(std_fitness))
        fitness_arr = np.append(fitness_arr, (avg_fitness, std_fitness))
        
        #collect initial avg and std-dev for each param over all particles--(meas. convergence)
        param_avgstd = measAvgStdDevforParticles(particles)
        avgstd_arr = np.append(avgstd_arr, param_avgstd)
        
    #save fitness array
    fitness_file = state_filedir + r"\fitness_list.npy"
    with open(fitness_file, 'wb') as fit_file:
        np.save(fit_file, fitness_arr)
        fit_file.close()
    
    #save array with dictionaries of param avg/std-dev over iterations
    param_perf_file = state_filedir + r"\param_perf_list.npy"
    with open(param_perf_file, 'wb') as perf_file:
        np.save(perf_file, avgstd_arr)
        perf_file.close()

    #quit simulation
    #robot.simulationQuit(1)
    #pause simulation
    robot.simulationSetMode(0)

    print("ending pitch chg experiment!!!")


if __name__ == '__main__':
    main()