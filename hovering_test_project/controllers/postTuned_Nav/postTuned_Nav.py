"""Using PSO-Tuned params, track traj.-error
1. Starting Postion-->(0, 0, 0.116)
2. Achieve take-off state to target altitude
3. Hover at target-altitude
4. Once vels==0, change yaw rotation to first waypoint
5. Chg-Yaw and pitch to move towards target
6. Move to add'l waypoints and collect traj. error
7. If not waypoints, land and stay grounded
"""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
###IMPORTS#######################################
import sys, os
import csv

from attr import s

from controller import Supervisor
from cmath import pi
from collections import deque

try: 
    import numpy as np
    from scipy import integrate
    from csv import DictWriter
    from simple_pid import PID
    from ahrs import DCM
    from ahrs.filters import Complementary
    from ahrs.common import quaternion as quat
except ImportError:
    sys.exit("Warning: 'numpy' module not found.")
#################################################


###Object-Class for Mavic Drone
class Mavic(Supervisor):

    #Mavic Constructor
    def __init__(self, waypoints, time_step, params, takeoff_threshold_velocity=1):
        #construct as supervisor
        Supervisor.__init__(self)
        try:
            self.time_step = time_step
        except:
            self.time_step = int(self.getBasicTimeStep())
        print("using timestep: {}...".format(self.time_step))

        #initiate drone states-->init-state=grounded
        self.drone_states = {'A': 'GNDD', #grounded
                       'B': 'TOFF', #takeoff
                       'C': 'HOVR', #hover
                       'D': 'OMWP', #land
                       'E': 'LAND' #land
                       } 
        #initiate drone state as grounded
        self.drone_state = self.drone_states['A']
        print("drone state init'd and grounded...")
        
        # Get and enable devices.
        self.camera = self.getDevice("camera")
        self.camera.enable(self.time_step)
        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)
        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)
        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.time_step)
        self.compass = self.getDevice("compass")
        self.compass.enable(self.time_step)
        self.accelerometer = self.getDevice("accelerometer")
        self.accelerometer.enable(self.time_step)

        #Get and enable motors
        self.front_left_motor = self.getDevice("front left propeller")
        self.front_right_motor = self.getDevice("front right propeller")
        self.rear_left_motor = self.getDevice("rear left propeller")
        self.rear_right_motor = self.getDevice("rear right propeller")
        self.camera_pitch_motor = self.getDevice("camera pitch")
        self.camera_pitch_motor.setPosition(0.7)
        motors = [self.front_left_motor, self.front_right_motor,
                  self.rear_left_motor, self.rear_right_motor]
        #set motor velocity (rad/s)
        for motor in motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(takeoff_threshold_velocity)

        #enable current pose for robot
        #self.current_pose = 6 * [0]  # X, Y, Z
        self.current_pose = self.gps.getValues()  # X, Y, Z, roll, pitch, yaw (from global)
        self.grnd_pos = self.current_pose

        #including previous-pose to calculate velocity
        self.previous_pose = 6 * [0]
        
        #arrays to sample acceleration to check for:
        # (x, y, z, rol, pit, yaw) steadiness-->vels == 0 
        self.xacc_arr = np.full((1, 40), 99.9)[0]
        self.yacc_arr = np.full((1, 40), 99.9)[0]
        self.altacc_arr = np.full((1, 40), 99.9)[0]
        self.rollvel_arr = np.full((1, 40), 99.9)[0]
        self.pitchvel_arr = np.full((1, 40), 99.9)[0]
        self.yawvel_arr = np.full((1, 40), 99.9)[0]
        self.time_interval_arr = np.full((1, 40), 99.9)[0]
        self.acc_idx = 0
        
        #init. setpoints for PID-Trajectory Controller
        self.xpos_setpoint = 0.0001#replace with initial waypoint
        self.ypos_setpoint = 0.0001#replace with initial waypoint
        #init PID controllers for xpos and ypos
        self.xposPID = PID(float(params["x_Kp"]), 
                          float(params["x_Ki"]),
                          float(params["x_Kd"]), 
                          setpoint=float(self.xpos_setpoint))
        self.yposPID = PID(float(params["y_Kp"]), 
                          float(params["y_Ki"]),
                          float(params["y_Kd"]), 
                          setpoint=float(self.ypos_setpoint))

        #init. setpoints for PID-Attidue Controller
        self.yaw_setpoint = 0
        self.pitch_setpoint = 0
        self.roll_setpoint = 0
        self.throttle_setpoint = 1
        #init PID controllers for roll, pitch, throttle, yaw (attitude)
        self.pitchPID = PID(float(params["pitch_Kp"]), 
                            float(params["pitch_Ki"]), 
                            float(params["pitch_Kd"]), 
                            setpoint=float(self.pitch_setpoint))
        self.rollPID = PID(float(params["roll_Kp"]), 
                           float(params["roll_Ki"]), 
                           float(params["roll_Kd"]), 
                           setpoint=float(self.roll_setpoint))
        self.throttlePID = PID(float(params["throttle_Kp"]), 
                               float(params["throttle_Ki"]), 
                               float(params["throttle_Kd"]), 
                               output_limits=(-50.0, 50.0),
                               setpoint=float(self.throttle_setpoint))
        self.yawPID = PID(float(params["yaw_Kp"]), 
                          float(params["yaw_Ki"]),  
                          float(params["yaw_Kd"]), 
                          output_limits=(-5.0, 5.0),
                          setpoint=float(self.yaw_setpoint))
        
        #target values
        #set target altitude from given z-waypoint
        try:
            #Drone moves to Takeoff state
            self.target_position = waypoints[0]
            print("Using first waypt: {}".format(self.target_position))
            if self.target_position[2] > 0.5:#if target waypt altitude is > .5 meters
                #set tgt-altitude (z-waypt)
                self.target_altitude = float(self.target_position[2])
                #set setpoint for xposPID to x-waypt
                self.xpos_setpoint = self.target_position[0]
                self.xposPID.setpoint = self.xpos_setpoint
                #set setpoint for yposPID to y-waypt
                self.ypos_setpoint = self.target_position[1]
                self.yposPID.setpoint = self.ypos_setpoint
                #set drone state to takeoff
                self.drone_state = self.drone_states['B']
                if self.drone_state == 'TOFF':
                    print("Drone set for tgt-altitude..")
                    print("...drone at takeoff state.")
            else:
                self.target_altitude = float(0)
                print("Altitude is less than 0.5...drone remains grounded.")
        except:
            #Drone remains grounded
            print("No given waypoints...")
            self.target_position = [0, 0, 0] #X, Y, Altitude(Z)
            print("...using starting point as target position")
            self.target_altitude = float(params["target_altitude"])
        #setting trottlePID setpoint to params
        self.throttlePID.setpoint = self.target_altitude
        print("Setting tgt-alt: {}".format(self.target_altitude))
        self.waypoint_index = 0
        self.target_alt_thresh = 0.2
        self.target_waypt_err = 0
        self.target_waypt_tol = 0.5
        self.tgt_heading_tol = 0.05
        self.hover_pos_set = False
        self.pitch_trav_lock = False
        self.yaw_trav_lock = False
        self.start_time = 0
        self.current_time = 0
        self.tgt_num_steps = 62.5 #if timestep=16ms, then 1s=62.5 steps
        self.tgt_time_intrvl = (self.getBasicTimeStep()/1000) * self.tgt_num_steps 
        self.check_yaw = False
        self.cut_motors = False
        self.sim_fin = False
        print("tgt_time_intvl: {}".format(self.tgt_time_intrvl))
        print("Mavic Drone initialized...")

    
    def clamp(self, value, value_min, value_max):
        return min(max(value, value_min), value_max)


    #set position for robot
    def set_position(self, pos):
        """
        Set the new absolute position of the robot
        Parameters:
            pos (list): [X,Y,Z] current absolute position and angles
        """
        self.current_pose = pos
        

    #set previous position of robot
    def set_previous_position(self, pos):
        self.previous_pose = pos

    
    #update attitude arrays per timestep
    def updateTransAccRotVelArrays(self, stable_arr):
        if self.acc_idx == 40:
            self.rollTransAccRotVelArrays()
            self.xacc_arr[-1] = stable_arr[0]
            self.yacc_arr[-1] = stable_arr[1]
            self.altacc_arr[-1] = stable_arr[2]
            self.rollvel_arr[-1] = stable_arr[3]
            self.pitchvel_arr[-1] = stable_arr[4]
            self.yawvel_arr[-1] = stable_arr[5]
            self.time_interval_arr[-1] = stable_arr[6]
        else:
            self.xacc_arr[self.acc_idx] = stable_arr[0]
            self.yacc_arr[self.acc_idx] = stable_arr[1]
            self.altacc_arr[self.acc_idx] = stable_arr[2]
            self.rollvel_arr[self.acc_idx] = stable_arr[3]
            self.pitchvel_arr[self.acc_idx] = stable_arr[4]
            self.yawvel_arr[self.acc_idx] = stable_arr[5]
            self.time_interval_arr[self.acc_idx] = stable_arr[6]
            self.acc_idx += 1
        

    #if acc_idx == size of arays, 
    #then-->roll the arrays to left and write val to last index 
    def rollTransAccRotVelArrays(self):
        self.xacc_arr = np.roll(self.xacc_arr, -1)
        self.yacc_arr = np.roll(self.yacc_arr, -1)
        self.altacc_arr = np.roll(self.altacc_arr, -1)
        self.rollvel_arr = np.roll(self.rollvel_arr, -1)
        self.pitchvel_arr = np.roll(self.pitchvel_arr, -1)
        self.yawvel_arr = np.roll(self.yawvel_arr, -1)
        self.time_interval_arr = np.roll(self.time_interval_arr, -1)


    #check for hover state: if current 20 samples == 0, for [ xacc, yacc, zacc]
    def isStable(self):
        #print("xacc: {}".format(np.round(abs(np.mean(self.xacc_arr)), 4)))
        #print("yacc: {}".format(np.round(abs(np.mean(self.yacc_arr)), 4)))
        #print("altacc: {}".format(np.round(abs(np.mean(self.altacc_arr)), 2)))
        #print("rollvel: {}".format(np.round(abs(np.mean(self.rollvel_arr)), 4)))
        #print("pitchvel: {}".format(np.round(abs(np.mean(self.pitchvel_arr)), 4)))
        #print("yawvel: {}".format(np.round(abs(np.mean(self.yawvel_arr)), 7)))
        

        stable_list = np.array([np.round(abs(np.mean(self.xacc_arr)), 4) <= 1e-2,#m/s^2
                                np.round(abs(np.mean(self.yacc_arr)), 4) <= 1e-2,#m/s^2
                                np.round(abs(np.mean(self.altacc_arr)), 2) >= 9.8 and 
                                np.round(abs(np.mean(self.altacc_arr)), 2) <= 9.9,#m/s^2
                                np.round(abs(np.mean(self.rollvel_arr)), 4) <= 1e-3,#rad/s
                                np.round(abs(np.mean(self.pitchvel_arr)), 4) <= 1e-3,#rad/s
                                np.round(abs(np.mean(self.yawvel_arr)), 4) <= 1e-2])#rad/s



        stable_dict = {'xacc': np.round(abs(np.mean(self.xacc_arr)), 4) <= 1e-2,
                        'yacc': np.round(abs(np.mean(self.yacc_arr)), 4) <= 1e-2,
                        'altacc': np.round(abs(np.mean(self.altacc_arr)), 2) >= 9.8 and 
                                  np.round(abs(np.mean(self.altacc_arr)), 2) <= 9.9,
                        'rollvel': np.round(abs(np.mean(self.rollvel_arr)), 4) <= 1e-3,
                        'pitchvel': np.round(abs(np.mean(self.pitchvel_arr)), 4) <= 1e-3,
                        'yawvel': np.round(abs(np.mean(self.yawvel_arr)), 4) <= 1e-2}
        #for key, val in stable_dict.items():
        #    print("{}: {}".format(key, val))
        #print("/n")
        return np.all(stable_list)


    #set initial tgt for pitch and roll at sim start
    def setTgtPitchAndRoll(self, x_pitch_val, y_roll_val):
        self.roll_setpoint = self.yposPID(y_roll_val)
        self.rollPID.setpoint = self.roll_setpoint
        print("roll setpoint changed to: {}".format(self.roll_setpoint))
            
        self.pitch_setpoint = self.xposPID(x_pitch_val)
        self.pitchPID.setpoint = self.pitch_setpoint
        print("pitch setpoint changed to: {}".format(self.pitch_setpoint))
        

    #set motor velocity to 0
    def cutMotors(self):
        self.cut_motors = True
        

    #when trigd, rotate bot to start-yaw, dec. altitude to 0.5m, cut motors
    def setYawAndLand(self, local2global_rot_matrix):
        #set tgt-heading to 0.00001
        self.tgt_yaw_heading = self.getTgtYawHeading() + self.current_pose[5]
        
        #collect bool if w/in tgt-heading
        yaw_chg_bool = abs(abs(self.tgt_yaw_heading) - abs(self.current_pose[5])) > self.tgt_heading_tol
        
        #initially from 'HOVR' stable and away from tgt yaw;
        #if (yaw_chg_bool_pos or yaw_chg_bool_neg) and self.hover_pos_set:
        if yaw_chg_bool and self.hover_pos_set:
            #print("w/o tolerance 1...........................................")
            self.hover_pos_set = False
            #self.yaw_trav_lock = False
            print("changing yaw...")
            #self.setTgtYawPIDheading()

        #else if w/in bounds of yaw-chg and stable and hover-pos unlockd
        elif not yaw_chg_bool and not self.hover_pos_set:
            #print("w/in tolerance..............................................")
            self.yaw_trav_lock = True
            if self.target_altitude > 0.15:
                self.target_altitude = self.target_altitude / 2.0
                if self.throttle_setpoint != self.target_altitude:
                    self.throttle_setpoint = self.target_altitude
                    self.throttlePID.setpoint = self.throttle_setpoint
            
        #else if away from tgt-Yaw and bot in flight;
        elif yaw_chg_bool and not self.hover_pos_set:
            #print("w/o tolerance 2..............................................")
            if self.check_yaw:
                self.setTgtYawPIDheading()





    #when triggered to go to waypoint
    def moveToTarget(self, local2global_rot_matrix):
        #collect tgt-heading to tgt-waypt
        self.tgt_yaw_heading = self.getTgtYawHeading() + self.current_pose[5]
        #self.tgt_yaw_heading = self.getTgtYawHeading()
        #print("yaw_set: {:.4f} tgt_yaw: {:.4f} yaw act: {:.4f} yaw_diff: {:.4f} tol: {}".format(self.yaw_setpoint, 
        #                                                                                       self.tgt_yaw_heading, 
        #                                                                                       self.current_pose[5], 
        #                                                                                       abs(abs(self.tgt_yaw_heading) - abs(self.current_pose[5])),
        #                                                                                       self.tgt_heading_tol))
        #print("isStable: {}".format(self.isStable()))
        #print("hover_pos_set: {}".format(self.hover_pos_set))
        #print("pitch-lock: {}".format(self.pitch_trav_lock))
        #print("yaw-lock: {}".format(self.yaw_trav_lock))
        
        #collect bool if w/in tgt-heading
        yaw_chg_bool = abs(abs(self.tgt_yaw_heading) - abs(self.current_pose[5])) > self.tgt_heading_tol
        #print("yaw_chg_bool+: {}".format(yaw_chg_bool_pos))
        #print("yaw_chg_bool-: {}".format(yaw_chg_bool_neg))
        #print("yaw_chg_bool: {}".format(yaw_chg_bool))

        #initially from 'HOVR' stable and away from tgt yaw;
        #if (yaw_chg_bool_pos or yaw_chg_bool_neg) and self.hover_pos_set:
        if yaw_chg_bool and self.hover_pos_set:
            #print("w/o tolerance 1...........................................")
            #then unlock hover_pos and yaw and chg-Yaw
            if self.pitch_trav_lock:# and self.yaw_trav_lock:
                self.hover_pos_set = False
                #self.yaw_trav_lock = False
                print("changing yaw...")
                #self.setTgtYawPIDheading()

        elif not yaw_chg_bool and self.hover_pos_set:
            if self.pitch_trav_lock:# and self.yaw_trav_lock:
                self.hover_pos_set = False
                #self.yaw_trav_lock = False
                print("yaw aligned w/ target waypt...")
                
        #else if w/in bounds of yaw-chg and stable and hover-pos unlockd
        elif not yaw_chg_bool and not self.hover_pos_set:
            #print("w/in tolerance..............................................")
            self.yaw_trav_lock = True
            #if yaw is unlokd, then lock yaw and unlok pitch and move bot fwd
            if self.pitch_trav_lock:# and self.isStable():# and self.yaw_trav_lock == False:
                print("moving forward...")
                self.chgPitch_to_target_v2(local2global_rot_matrix)
                self.pitch_trav_lock = False
            
        #else if away from tgt-Yaw and bot in flight;
        elif yaw_chg_bool and not self.hover_pos_set:
            #print("w/o tolerance 2..............................................")
            if self.check_yaw:
                self.setTgtYawPIDheading()
                self.pitch_trav_lock = True

        

        
        
    #return tgt-yaw-heading
    def getTgtYawHeading(self):
        #calculate yaw_setpoint for target waypoint
         # This will be in ]-pi;pi]
        yaw_diff = np.arctan2(self.target_position[1] - self.current_pose[1],
                              self.target_position[0] - self.current_pose[0])
        
        #This is now in ]-2pi;2pi[
        angle_left = yaw_diff - self.current_pose[5]
        #print("angle-left: {:.4f}".format(angle_left))

        #Normalize turn angle to ]-pi;pi]
        angle_left = (angle_left + 2 * np.pi) % (2 * np.pi)
        #print("norm angle-left: {:.4f}".format(angle_left))

        if (angle_left > np.pi):
            angle_left -= 2 * np.pi
            #print("new angle left: {}".format(angle_left))
        
        return angle_left
    
    #set tgt-yaw-heading
    def setTgtYawPIDheading(self):
        if self.yaw_setpoint != self.tgt_yaw_heading and self.drone_state == 'OMWP':
            #self.yaw_setpoint = self.tgt_yaw_heading
            self.yaw_setpoint = self.tgt_yaw_heading
            self.yawPID.setpoint = self.yaw_setpoint
            print("yaw changed to {:.4f}...".format(self.yaw_setpoint))
            
        
    #change pitch to move towards tgt-waypoint
    def chgPitch_to_target_v2(self, local2global_rot_matrix):
        print("in version 2")
        #collect Rot(G->L) * tgt-positions
        tgt_positions = np.array([self.target_position[0],
                                  self.target_position[1],
                                  self.target_position[2]])
        print("l2g_RotMat:\n {}".format(local2global_rot_matrix))
        local_tgt_setpts = np.matmul(local2global_rot_matrix, tgt_positions)
        print("locl tgt setpts:{}".format(local_tgt_setpts))
        
        #set xposPID setpoint
        #self.xpos_setpoint = -local_tgt_setpts[0]
        self.xpos_setpoint = -tgt_positions[0]
        self.xposPID.setpoint = self.xpos_setpoint
        
        #set yposPID setpoint
        #self.ypos_setpoint = local_tgt_setpts[1]
        self.ypos_setpoint = tgt_positions[1]
        self.yposPID.setpoint = self.ypos_setpoint


    #set bot to hover at current pos
    def setHoverPosition(self, local2global_rot_matrix):
        print("setting hover position...")
        curr_positions = np.array([self.current_pose[0],
                                   self.current_pose[1],
                                   self.current_pose[2]])
        print("l2g_RotMat:\n {}".format(local2global_rot_matrix))
        hover_pos_setpts = np.matmul(local2global_rot_matrix, curr_positions)
        print("locl tgt setpts:{}".format(hover_pos_setpts))
        
        #set xposPID setpoint
        self.xpos_setpoint = -hover_pos_setpts[0]
        self.xposPID.setpoint = self.xpos_setpoint
        
        #set yposPID setpoint
        self.ypos_setpoint = hover_pos_setpts[1]
        self.yposPID.setpoint = self.ypos_setpoint
        
        



    #run experiment for analyzing yaw change to waypoint
    def run(self, files_dict, waypoints, params):
        t1 = self.getTime()
        #calcd_time_step = (self.time_step * 4) / 1000
        calcd_time_step = self.time_step / 1000
        print("using calcd time step for PIDs: {}".format(calcd_time_step))
        
        #collect mass of quadrotor
        quad_mass = params['quadrotor_mass']
        print("quadrotor mass: {}kg".format(quad_mass))
        
        #collect world gravity
        world_grav = params['world_gravity']
        print("gravity: {}m/s^2".format(world_grav))
        
        #set sampling time for PIDs
        self.xposPID.sample_time = calcd_time_step
        self.yposPID.sample_time = calcd_time_step
        self.pitchPID.sample_time = calcd_time_step
        self.rollPID.sample_time = calcd_time_step
        self.yawPID.sample_time = calcd_time_step
        self.throttlePID.sample_time = calcd_time_step

        #self.xposPID.proportional_on_measurement = True
        #self.yposPID.proportional_on_measurement = True
        #self.pitchPID.proportional_on_measurement = True
        #self.rollPID.proportional_on_measurement = True
        #self.yawPID.proportional_on_measurement = True
        #self.throttlePID.proportional_on_measurement = True
        print("starting to run experiment...")

        #verify zpos waypoint is > 0.5
        takeoff_bool = False
        if self.target_altitude > 0.5:
            takeoff_bool = True
            
        #collect sim start time
        self.start_time = self.getTime()
            
        drone_bot = self.getSelf()
        self.grnd_pos = drone_bot.getPosition()
        print("Initial Position: ({}, {}, {})".format(self.grnd_pos[0],
                                                      self.grnd_pos[1],
                                                      self.grnd_pos[2]))
        
        #begin traj.-error exp., if target altitude given is > 0.5
        while self.step(self.time_step) != -1 and takeoff_bool:
            
            #collect current time from i-th step
            self.current_time = self.getTime()
            #check for tgt-time-interval
            if int(self.current_time - self.start_time) >= self.tgt_time_intrvl:
                #print("start time: {} current time: {}".format(self.start_time, self.current_time))
                self.start_time = self.current_time
                self.check_yaw = True
            else:
                self.check_yaw = False

            #collect orientation of drone relative to global (ENU)
            drone_rot = drone_bot.getOrientation()
            #i = 0
            #print("drone rot:")
            #while i < len(drone_rot):
            #    print("[{:.4f}, {:.4f}, {:.4f}]".format(drone_rot[i],
            #                                drone_rot[i+1],
            #                                drone_rot[i+2]))
            #    i += 3
            
            #collect current attitude
            roll = self.imu.getRollPitchYaw()[0]# + pi / 2.0
            pitch = self.imu.getRollPitchYaw()[1]
            #yaw = self.compass.getValues()[0]
            yaw = self.imu.getRollPitchYaw()[2]
            #collect  current position
            x_pos, y_pos, altitude = self.gps.getValues()
            roll_vel, pitch_vel, yaw_vel = self.gyro.getValues()
            #print(self.gyro.getLookupTable())
            self.set_position([x_pos, y_pos, altitude, roll, pitch, yaw])

            #collect velocity
            #x_vel, y_vel, alt_vel = self.getVelocities(calcd_time_step)
            gps_velocity = self.gps.getSpeedVector()
            x_vel = gps_velocity[0]
            y_vel = gps_velocity[1]
            alt_vel = gps_velocity[2]
            self.previous_pose = self.current_pose
            
            #collect acceleration
            x_acc, y_acc, alt_acc = self.accelerometer.getValues()
            
            #arr used to hold accelerations to determine stability
            stable_arr = np.array([x_acc, y_acc, alt_acc, 
                                   roll_vel, pitch_vel, yaw_vel, 
                                   self.getTime()])
            self.updateTransAccRotVelArrays(stable_arr)
            
            #collect bot to tgt-waypt error
            self.target_waypt_err = np.sqrt(((self.target_position[0] - self.current_pose[0]) ** 2) + ((self.target_position[1] - self.current_pose[1]) ** 2))
            #if self.target_waypt_err < self.target_waypt_tol:
            #    print("tgt_waypt err: {}".format(self.target_waypt_err))
            
            #write mavic2pro state
            mavic_state = [x_pos, y_pos, altitude, 
                           roll, pitch, yaw,
                           x_vel, y_vel, alt_vel,
                           x_acc, y_acc, alt_acc,
                           roll_vel, pitch_vel, yaw_vel,
                           self.getTime(),
                           self.target_waypt_err]
            writeMavicState(mavic_state, files_dict['state-file'])
            
            #collect local2GlobalRotMatrix
            #local2global_rot_matrix = DCM(rpy=[roll, pitch, yaw])
            local2global_rot_matrix = DCM(z=yaw)
            
            #check roll and pitch setpoints: if they're 0-->set as output of (x,y)posPIDs
            if self.roll_setpoint == 0 and self.pitch_setpoint == 0:
                self.setTgtPitchAndRoll(-x_pos, y_pos)
                
                
            #flags to enter Hover state from Takeoff State
            toff_hover_bool = [altitude < self.target_altitude + self.target_alt_thresh,
                               altitude > self.target_altitude - self.target_alt_thresh,
                               self.isStable(),
                               self.drone_state == 'TOFF']
            omwp_hover_bool = [altitude >= self.target_altitude - self.target_alt_thresh,
                               abs(self.target_waypt_err) < self.target_waypt_tol,
                               #self.isStable(),
                               self.drone_state == 'OMWP']
            omwp_bool = [self.drone_state == 'OMWP']
            land_bool = [self.drone_state == 'LAND']
        
            #if bot is in Takeoff state
            if np.all(toff_hover_bool) or np.all(omwp_hover_bool):
                self.setHoverPosition(local2global_rot_matrix)
                self.drone_state = self.drone_states['C']
                if self.drone_state == "HOVR":
                    self.pitch_trav_lock = True
                    self.yaw_trav_lock = True
                    self.hover_pos_set = True
                    print("in hover state, awaiting waypt...")
                #if not at the end of the list of waypoints
                if self.isStable and self.waypoint_index != len(waypoints) - 1:
                    self.waypoint_index += 1
                    #change yaw to move to first waypoint
                    self.target_position = waypoints[self.waypoint_index]
                    print("current target: {}".format(self.target_position))
                    ###move bot towards tgt
                    self.drone_state = self.drone_states['D']
                    if self.drone_state == 'OMWP':
                        print("on the move to waypt state...")
                    self.moveToTarget(local2global_rot_matrix)
                else:
                    print("Reached end of waypts...")
                    self.setHoverPosition(local2global_rot_matrix)
                    self.hover_pos_set = True
                    self.drone_state = self.drone_states['E']
                    if self.drone_state == 'LAND':
                        print("in landing state...")
                    
            #if bot is On the Move to WayPt state
            elif np.all(omwp_bool):
                #print("in omwp bool...")
                self.moveToTarget(local2global_rot_matrix)
            elif np.all(land_bool):
                self.sim_fin = True
                #if altitude < 0.15:
                #    self.cutMotors()
                #    self.drone_state = self.drone_states['A']
                #    if self.drone_state == 'GNDD':
                #        print("drone is grounded...")
                #        #self.sim_fin = True
                #else:
                #    self.setYawAndLand(local2global_rot_matrix)

        
            #collect inputs
            #transform xpos and ypos depending on yaw
            # Modify xposPID and yposPID based on yaw angle
            mod_xposPID = self.xposPID(-x_pos) * np.cos(yaw) - self.yposPID(y_pos) * np.sin(yaw)
            mod_yposPID = self.xposPID(-x_pos) * np.sin(yaw) + self.yposPID(y_pos) * np.cos(yaw)


            #roll
            #roll_input = float(params["k_roll_p"]) * self.clamp(roll, -1, 1) + roll_vel + self.rollPID(roll) + self.clamp(self.yposPID(y_pos), -1, 1)
            roll_input = float(params["k_roll_p"]) * self.clamp(roll, -1, 1) + roll_vel + self.rollPID(roll) + self.clamp(mod_yposPID, -1, 1)
            #pitch
            #pitch_input = float(params["k_pitch_p"]) * self.clamp(pitch, -1, 1) + pitch_vel + self.pitchPID(pitch) + self.clamp(self.xposPID(-x_pos), -1, 1)
            pitch_input = float(params["k_pitch_p"]) * self.clamp(pitch, -1, 1) + pitch_vel + self.pitchPID(pitch) + self.clamp(mod_xposPID, -1, 1)



            #yaw
            yaw_input = self.clamp(self.yawPID(yaw), -1, 1)
            #yaw_input = self.yawPID(yaw_ang)
            #yaw_input = self.yawDotPID(yaw_vel)
            #thrust
            #print("climbing to target altitude: {:.4f}".format(float(self.target_altitude)))
            diff_altitude = self.target_altitude - altitude + float(params["k_vertical_offset"])
            #print("diff_alt: {:.4f}, {}".format(diff_altitude, type(diff_altitude)))
            clamped_difference_altitude = self.clamp(self.target_altitude - altitude + float(params["k_vertical_offset"]), -1.0, 1.0)
            #print("clamp_diff: {:.4f}, {}".format(clamped_difference_altitude, type(clamped_difference_altitude)))
            #vertical_input = self.throttlePID(clamped_difference_altitude )
            #vertical_input = self.throttlePID(altitude)
            vertical_input = self.throttlePID(altitude)
            #vertical_input = (1 / (np.cos(roll_ang) * np.cos(pitch_ang))) * (self.throttlePID(zpos_global) + (float(quad_mass) * float(world_grav)))
            #print("inputs-->roll: {:.4f}, pitch: {:.4f}, yaw: {:.4f}, vertical: {:.4f}".format(roll_input, pitch_input, yaw_input, vertical_input))

            ##Pause simulation when conditions are met
            #set exit-condition for time #pause after 10min
            time_bool = False
            if self.getTime() > float(1200):
                time_bool = True
                print("sim timed out...")
                
            #set exit-conditions for position
            #--->2 conditions: if bot is in hover->stay at curr_waypt
            #----------------->if bot is in pitch-chg state->swap pos cond with angle_tol
            wypnt_tolerance = float(20)
            position_cond = np.array([x_pos, y_pos, altitude])
            position_bool = False
            #only checking for going above target altitude
            if (position_cond[2] > self.target_altitude + (wypnt_tolerance * 5)):#for altitude
                position_bool = True
                print("z became true...")
            #check x and y position cond
            if position_cond[0] > self.target_position[0] + wypnt_tolerance or position_cond[0] < self.target_position[0] - wypnt_tolerance:
                position_bool = True
                print("x became true...")
            if position_cond[1] > self.target_position[1] + wypnt_tolerance or position_cond[1] < self.target_position[1] - wypnt_tolerance:
                position_bool = True
                print("y became true...")

            #set exit-conditions for attitude
            attitude_cond_r_p = np.array([roll, pitch])#roll and pitch cond
            attitude_cond_y = np.array([yaw])#yaw cond
            attitude_bool = False
            #only checking for roll and pitch since we change yaw
            for att in attitude_cond_r_p:
                if (att > 1.25) or (att < -1.25):
                    attitude_bool = True
                    print("drone lost attitude...")
                    
            #set exit-conditions for velocity
            alt_bool = False
            if alt_vel < -float(8):
                alt_bool = True
                print("alt-vel became true...")
                    
            #motor inputs
            front_left_motor_input = float(params["k_vertical_thrust"]) + vertical_input - yaw_input + pitch_input - roll_input
            front_right_motor_input = float(params["k_vertical_thrust"]) + vertical_input + yaw_input + pitch_input + roll_input
            rear_left_motor_input = float(params["k_vertical_thrust"]) + vertical_input + yaw_input - pitch_input - roll_input
            rear_right_motor_input = float(params["k_vertical_thrust"]) + vertical_input - yaw_input - pitch_input + roll_input
            #print("motors:\n{:.4f}, {:.4f}, {:.4f}, {:.4f}".format(front_left_motor_input, front_right_motor_input, rear_left_motor_input, rear_right_motor_input))

            #write PID and rotational and vertical inputs
            pid_inputs = [self.xposPID(-x_pos), self.yposPID(y_pos), 
                          self.rollPID(roll), self.pitchPID(pitch), 
                          self.yawPID(yaw), self.throttlePID(altitude),
                          roll_input, pitch_input, 
                          yaw_input, vertical_input,
                          diff_altitude, clamped_difference_altitude,
                          front_left_motor_input, front_right_motor_input,
                          rear_left_motor_input, rear_right_motor_input]
            writePIDandInputs(pid_inputs, files_dict['pid-file'])
            
            #set exit-conditions for motor inputs
            mtr_input_cond = np.array([front_left_motor_input, 
                                        front_right_motor_input, 
                                        rear_left_motor_input,
                                        rear_right_motor_input])
            #motor input pause condition
            mtr_inpt_bool = False
            for i, inpt in enumerate(mtr_input_cond):
                if inpt > 300:
                    mtr_inpt_bool = True
                    print("{}th motor at {}....".format(i, mtr_inpt_bool))

            #simulation pause conditions
            if attitude_bool or mtr_inpt_bool or position_bool or time_bool or alt_bool or self.sim_fin:
                self.simulationSetMode(0)

            #set motor velocity
            self.front_left_motor.setVelocity(front_left_motor_input)
            self.front_right_motor.setVelocity(-front_right_motor_input)
            self.rear_left_motor.setVelocity(-rear_left_motor_input)
            self.rear_right_motor.setVelocity(rear_right_motor_input)




#collect waypoints, from waypoint_logfile.txt
def collectWaypoints() -> list:
    waypoint_list = []
    filename = os.getcwd() + r"\waypoint_logfile.txt"
    #filename = os.getcwd() + r"\waypoint_logfile_infinity.txt"
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            points = line.strip().split(" ")
            waypoint_list.append([float(points[0]), float(points[1]), float(points[2])])
    return waypoint_list




#clear file, if exists
def clearFileIfExists(filename):
    if os.path.isfile(filename):
        os.remove(filename)
        

#write state of Drone every timestep
def writeMavicState(input_arr, filename):

    field_names = ['x_pos', 'y_pos', 'z_pos', 
                   'roll_rot', 'pitch_rot', 'yaw_rot', 
                   'x_vel', 'y_vel', 'alt_vel',
                   'x_acc', 'y_acc', 'alt_acc',
                   'roll_vel', 'pitch_vel', 'yaw_vel',
                   'timestep',
                   'tgt_waypt_err']

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
                'tgt_waypt_err': input_arr[16]}

    with open(filename, 'a', newline='', encoding='utf-8') as f_obj:
        dictWriter_obj = DictWriter(f_obj, fieldnames=field_names)
        try:
            dictWriter_obj.writerow(csv_dict)
        except:
            print("sim crashed on writing state...")
        f_obj.close()


#Document PID and Inputs per timestep
def writePIDandInputs(input_arr, filename):

    field_names = ['xposPID', 'yposPID', 
                   'rollPID', 'pitchPID', 
                   'yawPID', 'throttlePID',
                   'roll_input', 'pitch_input', 
                   'yaw_input', 'vertical_input', 
                   'diff_altitude', 'clampd_diff_altitude',
                   'front_left_motor', 'front_right_motor',
                   'rear_left_motor', 'rear_right_motor']

    csv_dict = {'xposPID': input_arr[0],
                'yposPID': input_arr[1],
                'rollPID': input_arr[2],
                'pitchPID': input_arr[3],
                'yawPID': input_arr[4],
                'throttlePID': input_arr[5],
                'roll_input': input_arr[6],
                'pitch_input': input_arr[7],
                'yaw_input': input_arr[8],
                'vertical_input': input_arr[9],
                'diff_altitude': input_arr[10],
                'clampd_diff_altitude': input_arr[11],
                'front_left_motor': input_arr[12],
                'front_right_motor': input_arr[13],
                'rear_left_motor': input_arr[14],
                'rear_right_motor': input_arr[15]}

    with open(filename, 'a') as f_obj:
        dictWriter_obj = DictWriter(f_obj, fieldnames=field_names)
        try:
            dictWriter_obj.writerow(csv_dict)
        except:
            print("sim crashed on writing PID/input...")
        f_obj.close()


def main():
    #clear output files if they exit
    filedir = r"C:\Users\ericx\OneDrive\Desktop\mine\grad_school\MARS_Lab_Swarm\Gazebo_dragonFly\webots_tutorial\hovering_test_project\python_utilities"
    ###experimental-designator filename
    #state_filename = filedir + r"\mavic_state6_wDefault.csv"
    state_filename = filedir + r"\mavic_state6_wTuned.csv"
    #pid_filename = filedir + r"\PID_and_inputs6_wDefault.csv"
    pid_filename = filedir + r"\PID_and_inputs6_wTuned.csv"
    clearFileIfExists(state_filename)
    clearFileIfExists(pid_filename)
    files_dict = {'state-file': state_filename, 
                  'pid-file': pid_filename}
    
    #collect parameters for PID-experiment
    print("numpy version: {}".format(np.__version__))
    params = dict()

    with open("best_tuned_params_edit.csv", "r") as f:
    #with open("tuned_params_edit.csv", "r") as f:
    #with open("default_params_edit.csv", "r") as f:
        lines = csv.reader(f)
        for line in lines:
            #print(line)
            params[line[0]] = line[1]
    for key, val in params.items():
        print("{}: {}".format(key, val))
        
    #collect waypoints
    waypoints = collectWaypoints()
    print(waypoints)
    
    #time-step=8ms
    TIME_STEP = int(params["QUADCOPTER_TIME_STEP"])
    #takeoff-thresh-vel = 160
    TAKEOFF_THRESHOLD_VELOCITY = int(params["TAKEOFF_THRESHOLD_VELOCITY"])
    
    #build robot
    robot = Mavic(waypoints=waypoints,
                  time_step=TIME_STEP,
                  params=params,
                  takeoff_threshold_velocity=TAKEOFF_THRESHOLD_VELOCITY)
    #run robot experiment
    robot.run(files_dict,
              waypoints,
              params=params)




if __name__ == "__main__":
    main()