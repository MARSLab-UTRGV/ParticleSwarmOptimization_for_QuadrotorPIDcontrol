"""PID experimental controller for PSO.
1. Start PSO for Pitch PID-params
2. Params Domain: 
    {'P': (0,2)
    {'I': (0.00001, 0.01)
    {'D': (0, 1)
3. PSO:
    pbest_obj-fcn = use MSE over reference traj and exp. traj
    X = {PID parameters} --(maybe try to input tgt-altitude)
    V = initial set to same-dims as X, all 0s
    V_update = wgt*V + c1*r1*(pbest -X) + c2*r2*(gbest -X)
"""


# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
###IMPORTS#######################################
import sys, os, time
import csv
import struct
from controller import Robot, Emitter, Receiver
from cmath import pi

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
class Mavic(Robot):

    def __init__(self, waypoints, time_step, params, takeoff_threshold_velocity=1):
        Robot.__init__(self)

        try:
            self.time_step = time_step
        except:
            self.time_step = int(self.getBasicTimeStep())
        print("mavic using timestep: {}".format(self.time_step))

        #index for waypts
        self.waypoint_idx = 0
        self.curr_waypt = [0, 0, 0]
        self.next_waypt = [0, 0, 0]

        #initiate drone states-->init-state=grounded
        self.drone_states = {'A': 'grnd',
                       'B': 'toff',
                       'C': 'hovr',
                       'D': 'yawd',
                       'E': 'pitd',
                       'F': 'land',
                       'X': 'unks'}
        #initiate drone state as grounded
        self.drone_state = self.drone_states['A']
        #print("drone state init'd and grounded...")

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
        #include emitter and receiver to get comms with Supervisor
        self.mvc_rcvr = self.getDevice("mavic_rcvr")
        self.mvc_rcvr.enable(self.time_step)
        self.mvc_emtr = self.getDevice("mavic_emtr")

        #Get and enable motors
        self.front_left_motor = self.getDevice("front left propeller")
        self.front_right_motor = self.getDevice("front right propeller")
        self.rear_left_motor = self.getDevice("rear left propeller")
        self.rear_right_motor = self.getDevice("rear right propeller")
        self.camera_pitch_motor = self.getDevice("camera pitch")
        self.camera_pitch_motor.setPosition(0.7)
        motors = [self.front_left_motor, self.front_right_motor,
                  self.rear_left_motor, self.rear_right_motor]
        #set motor velocity
        for motor in motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(takeoff_threshold_velocity)

        #enable current pose for robot
        #self.current_pose = 6 * [0]  # X, Y, Z
        self.current_pose = self.gps.getValues()  # X, Y, Z, roll, pitch, yaw

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

        #init. setpoints for PD-Trajectory Controller
        self.xpos_setpoint = 0.0001#replace with initial waypoint
        self.ypos_setpoint = 0.0001#replace with initial waypoint
        #init PD controllers for xpos and ypos
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
            #print("Using first waypt: {}".format(self.target_position))
            if self.target_position[2] > 0.5:
                #set tgt-altitude (z-waypt)
                self.target_altitude = float(self.target_position[2])
                #set setpoint for xposPD to x-waypt
                self.xpos_setpoint = self.target_position[0]
                self.xposPID.setpoint = self.xpos_setpoint
                #set setpoint for yposPD to y-waypt
                self.ypos_setpoint = self.target_position[1]
                self.yposPID.setpoint = self.ypos_setpoint
                #set drone state to takeoff
                self.drone_state = self.drone_states['B']
                if self.drone_state == 'toff':
                    print("Drone set for tgt-altitude..")
                    print("...drone at takeoff state.")
            else:
                self.target_altitude = float(0)
                print("Altitude is less than 0.5...")
                print("...drone remains grounded.")
        except:
            #Drone remains grounded
            print("No given waypoints...")
            self.target_position = [0, 0, 0] #X, Y, Altitude(Z)
            print("...using starting point as target position")
            self.target_altitude = float(params["target_altitude"])
        #setting trottlePID setpoint to params
        self.throttlePID.setpoint = self.target_altitude
        print("Setting tgt-alt: {}".format(self.target_altitude))
        self.waypoint_idx = 0
        self.target_alt_thresh = 0.2
        self.target_waypt_err = 0
        self.target_waypt_tol = 0.01
        self.tgt_yaw_heading = 0
        self.tgt_heading_tol = 0.01
        self.sim_fin = False
        #print("Mavic Drone initialized...")

        
    #clamp-fcn: return min-val b/w input max value
    #and (the max b/w curr val and min val)
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
        #print("roll setpoint changed to: {}".format(self.roll_setpoint))
            
        self.pitch_setpoint = self.xposPID(x_pitch_val)
        self.pitchPID.setpoint = self.pitch_setpoint
        #print("pitch setpoint changed to: {}".format(self.pitch_setpoint))


    #change yaw to move to first waypoint
    def chgYaw_to_target(self, tgt_waypt):
        self.target_position = tgt_waypt
        print("current tgt-waypt: {}".format(self.target_position))

        #calculate yaw_setpoint for target waypoint
         # This will be in ]-pi;pi]
        yaw_chg = np.arctan2(self.target_position[1] - self.current_pose[1],
                             self.target_position[0] - self.current_pose[0])
        print("current yaw: {:.4f}".format(self.current_pose[5]))
        if self.tgt_yaw_heading == 0:
            self.tgt_yaw_heading = yaw_chg
        print("calcd yaw chg: {:.4f}".format(self.tgt_yaw_heading))

        # This is now in ]-2pi;2pi[
        angle_left = yaw_chg - self.current_pose[5]
        print("angle-left: {:.4f}".format(angle_left))

        # Normalize turn angle to ]-pi;pi]
        angle_left = (angle_left + 2 * np.pi) % (2 * np.pi)
        print("norm angle-left: {:.4f}".format(angle_left))

        if (angle_left > np.pi):
            angle_left -= 2 * np.pi
            print("new angle left: {}".format(angle_left))
        print("pitch-try: {:.4f}".format(np.log10(abs(angle_left))))

        if self.yaw_setpoint != angle_left and self.drone_state == "hovr":
            self.yaw_setpoint = angle_left
            self.yawPID.setpoint = self.yaw_setpoint
            self.drone_state = self.drone_states['D']
            if self.drone_state == "yawd":
                print("yaw set and in yaw-chgd state...")


    #change pitch to move towards tgt-waypoint
    def chgPitch_to_target_v2(self, local2global_rot_matrix):
        #print("in version 2")
        #collect Rot(G->L) * tgt-positions
        tgt_positions = np.array([self.target_position[0],
                                  self.target_position[1],
                                  self.target_position[2]])
        #print("l2g_RotMat:\n {}".format(local2global_rot_matrix))
        local_tgt_setpts = np.matmul(local2global_rot_matrix, tgt_positions)
        #print("locl tgt setpts:{}".format(local_tgt_setpts))
        
        #set xposPD setpoint
        #self.xpos_setpoint = -local_tgt_setpts[0]
        self.xpos_setpoint = -tgt_positions[0]
        self.xposPID.setpoint = self.xpos_setpoint
        
        #set yposPD setpoint
        #self.ypos_setpoint = local_tgt_setpts[1]
        self.ypos_setpoint = tgt_positions[1]
        self.yposPID.setpoint = self.ypos_setpoint


    #return true if drone is within target distance 
    def withinTgtDist(self):
        #withinBnds = [abs(self.target_position[0] - self.current_pose[0]) < self.target_waypt_tol,
         #             abs(self.target_position[1] - self.current_pose[1]) < self.target_waypt_tol]
        #print("current waypt err: {}".format(self.target_waypt_err))
        return self.target_waypt_err < self.target_waypt_tol


    #return positional velocities of robot
    def getVelocities(self, timestep):
        x_vel = (self.current_pose[0] - self.previous_pose[0]) / timestep
        y_vel = (self.current_pose[1] - self.previous_pose[1]) / timestep
        z_vel = (self.current_pose[2] - self.previous_pose[2]) / timestep
        #print("vels: {:.4f}, {:.4f}, {:.4f}".format(self.current_pose[0], self.previous_pose[0], timestep))
        #print("altitude vel: {:.4f}".format(z_vel))
        return x_vel, y_vel, z_vel


    #running quadrotor with given params
    def run(self, params, waypoints):
        t1 = self.getTime()
        #calcd_time_step = (self.time_step * 4) / 1000
        calcd_time_step = self.time_step / 1000
        #print("using calcd time step for PIDs: {}".format(calcd_time_step))
        
        #collect mass of quadrotor
        quad_mass = params['quadrotor_mass']
        #print("quadrotor mass: {}kg".format(quad_mass))
        
        #collect world gravity
        world_grav = params['world_gravity']
        #print("gravity: {}m/s^2".format(world_grav))

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
        print("starting run with parameters...")

        #verify zpos waypoint is > 0.5
        takeoff_bool = False
        if self.target_altitude > 0.5:
            takeoff_bool = True

        self.curr_waypt = waypoints[self.waypoint_idx]
        self.next_waypt = waypoints[self.waypoint_idx]
        
        print("bot set altitude: {}m".format(self.target_altitude))
        while self.step(self.time_step) != -1 and takeoff_bool:
            #Read sensors and send mavic2pro state to supervisor
            #pos_msg
            x_pos, y_pos, altitude = self.gps.getValues()
            pos_msg = struct.pack("ddd", x_pos, y_pos, altitude)
            self.mvc_emtr.send(pos_msg)

            #attitude_msg
            roll = self.imu.getRollPitchYaw()[0]
            pitch = self.imu.getRollPitchYaw()[1]
            yaw = self.imu.getRollPitchYaw()[2]
            att_msg = struct.pack("ddd", roll, pitch, yaw)
            self.mvc_emtr.send(att_msg)
            #set position and attitude
            self.set_position([x_pos, y_pos, altitude, roll, pitch, yaw])
            
            #collect rotational matrix
            local2global_rot_matrix = DCM(z=yaw)

            #velocity_msg
            gps_velocity = self.gps.getSpeedVector()
            x_vel = gps_velocity[0]
            y_vel = gps_velocity[1]
            alt_vel = gps_velocity[2]
            self.previous_pose = self.current_pose
            vel_msg = struct.pack("ddd", x_vel, y_vel, alt_vel)
            self.mvc_emtr.send(vel_msg)
            
            #position-acceleration msg
            x_acc, y_acc, alt_acc = self.accelerometer.getValues()
            pos_acc_msg = struct.pack("ddd", x_acc, y_acc, alt_acc)
            self.mvc_emtr.send(pos_acc_msg)
            
            #attitude-rotational-velocity msg
            roll_vel, pitch_vel, yaw_vel = self.gyro.getValues()
            acc_msg = struct.pack("ddd", roll_vel, pitch_vel, yaw_vel)
            self.mvc_emtr.send(acc_msg)
            
            #time_msg
            time_msg = struct.pack("d", self.getTime())
            self.mvc_emtr.send(time_msg)
            
            #arr used to hold accelerations to determine stability
            stable_arr = np.array([x_acc, y_acc, alt_acc, 
                                   roll_vel, pitch_vel, yaw_vel, 
                                   self.getTime()])
            self.updateTransAccRotVelArrays(stable_arr)
            
            #check roll and pitch setpoints: if they're 0-->set as output of (x,y)posPDs
            if self.roll_setpoint == 0 and self.pitch_setpoint == 0:
                self.setTgtPitchAndRoll(-x_pos, y_pos)
                
            #collect bot to tgt-waypt error
            self.target_waypt_err = np.sqrt(((self.target_position[0] - self.current_pose[0]) ** 2) + ((self.target_position[1] - self.current_pose[1]) ** 2))
    

            ##insert flags and flag-commands
            ##as drone moves thru each state
            #flags to enter Hover state from Takeoff State
            hover_bool = [altitude >= self.target_altitude - self.target_alt_thresh,
                          self.isStable(),
                          self.drone_state == 'toff']
            #flags to enter YawChange state from Hover state
            yaw_bool = [self.drone_state == "yawd",
                        self.isStable(),
                        self.tgt_yaw_heading != 0]
            #flags to enter PitchChange state from YawChangeState
            pitch_bool = [self.drone_state == "pitd",
                          self.isStable()]
            #flag to enter land-state and finish experiment
            end_bool = [self.drone_state == "land"]

            #if bot is in Takeoff state
            if np.all(hover_bool):
                self.drone_state = self.drone_states['C']
                if self.drone_state == "hovr":
                    print("in Hover state and changing yaw")
                if self.getTime() - t1 > 0.1 and self.waypoint_idx != len(waypoints):
                    self.waypoint_idx += 1
                    self.next_waypt = waypoints[self.waypoint_idx]
                    self.chgYaw_to_target(self.next_waypt)
            #else if bot is in the yaw change state 
            elif np.all(yaw_bool):
                if abs(self.tgt_yaw_heading - self.current_pose[5]) < self.tgt_heading_tol:
                    self.drone_state = self.drone_states['E']
                    print("drone in pitch-chg state...")
                #else:
                #    print("drone changing heading...")
            elif np.all(pitch_bool):
                if self.drone_state != self.drone_states['F']:
                    self.chgPitch_to_target_v2(local2global_rot_matrix)
                    self.drone_state = self.drone_states['F']
            elif np.all(end_bool):
                #print("reached last state...")
                if self.withinTgtDist() and self.isStable():
                    self.sim_fin = True

            #state_msg
            state_msg = struct.pack("4s", self.drone_state.encode('utf-8'))
            self.mvc_emtr.send(state_msg)

            #collect inputs
            #transform xpos and ypos depending on yaw
            # Modify xposPD and yposPD based on yaw angle
            #xpos-local rotated about z-axis-global
            mod_xposPD = self.xposPID(-x_pos) * np.cos(yaw) - self.yposPID(y_pos) * np.sin(yaw)
            #ypos-local rotated about z-axis-global
            mod_yposPD = self.xposPID(-x_pos) * np.sin(yaw) + self.yposPID(y_pos) * np.cos(yaw)
            #roll
            roll_input = float(params["k_roll_p"]) * self.clamp(roll, -1, 1) + roll_vel + self.rollPID(roll) + self.clamp(mod_yposPD, -1, 1)
            #pitch
            pitch_input = float(params["k_pitch_p"]) * self.clamp(pitch, -1, 1) + pitch_vel + self.pitchPID(pitch) + self.clamp(mod_xposPD, -1, 1)
            #yaw
            yaw_input = self.clamp(self.yawPID(yaw), -1, 1)
            #thrust
            diff_altitude = self.target_altitude - altitude + float(params["k_vertical_offset"])
            #print("diff_alt: {:.4f}, {}".format(diff_altitude, type(diff_altitude)))
            clamped_difference_altitude = self.clamp(self.target_altitude - altitude + float(params["k_vertical_offset"]), -1.0, 1.0)
            vertical_input = self.throttlePID(altitude)
            #print("inputs-->roll: {:.4f}, pitch: {:.4f}, yaw: {:.4f}, vertical: {:.4f}".format(roll_input, pitch_input, yaw_input, vertical_input))
                    
            #motor inputs
            front_left_motor_input = float(params["k_vertical_thrust"]) + vertical_input - yaw_input + pitch_input - roll_input
            front_right_motor_input = float(params["k_vertical_thrust"]) + vertical_input + yaw_input + pitch_input + roll_input
            rear_left_motor_input = float(params["k_vertical_thrust"]) + vertical_input + yaw_input - pitch_input - roll_input
            rear_right_motor_input = float(params["k_vertical_thrust"]) + vertical_input - yaw_input - pitch_input + roll_input
            #print("motors:\n{:.4f}, {:.4f}, {:.4f}, {:.4f}".format(front_left_motor_input, front_right_motor_input, rear_left_motor_input, rear_right_motor_input))
            
            #motor-msg (FL, FR, RL, RR)
            front_mtr_msg = struct.pack("dd", front_left_motor_input, front_right_motor_input)
            self.mvc_emtr.send(front_mtr_msg)
            rear_mtr_msg = struct.pack("dd", rear_left_motor_input, rear_right_motor_input)
            self.mvc_emtr.send(rear_mtr_msg)
            
            #posPID_msg (x, y)
            posPID_msg = struct.pack("dd", self.xposPID(x_pos), self.yposPID(y_pos))
            self.mvc_emtr.send(posPID_msg)
            #attPID_msg
            roll_ptch_msg = struct.pack("dd", self.rollPID(roll), self.pitchPID(pitch))
            self.mvc_emtr.send(roll_ptch_msg)
            yaw_throttle_msg = struct.pack("dd", self.yawPID(yaw), self.throttlePID(clamped_difference_altitude))
            self.mvc_emtr.send(yaw_throttle_msg)
            
            #input_msgs(roll, pitch, yaw, altitude) and diff altitude
            roll_ptch_input_msg = struct.pack("dd", roll_input, pitch_input)
            self.mvc_emtr.send(roll_ptch_input_msg)
            yaw_throttle_input_msg = struct.pack("dd", yaw_input, vertical_input)
            self.mvc_emtr.send(yaw_throttle_input_msg)
            diff_alt_msg = struct.pack("dd", diff_altitude, clamped_difference_altitude)
            self.mvc_emtr.send(diff_alt_msg)

            #set rotational velocity for motors
            self.front_left_motor.setVelocity(front_left_motor_input)
            self.front_right_motor.setVelocity(-front_right_motor_input)
            self.rear_left_motor.setVelocity(-rear_left_motor_input)
            self.rear_right_motor.setVelocity(rear_right_motor_input)





#collect waypoints, from waypoint_logfile.txt
def collectWaypoints() -> list:
    waypoint_list = []
    filename = os.getcwd() + r"\waypoint_logfile.txt"
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            points = line.strip().split(" ")
            waypoint_list.append([float(points[0]), float(points[1]), float(points[2])])
    return waypoint_list


#retreive params from particle csvfile
def getParams(params_file):
    params_dict = dict()
    while not os.path.exists(params_file):
        print("waiting for file to be written...")
        time.sleep(1)

    if os.path.isfile(params_file):
        with open(params_file, "r") as f:
            lines = csv.reader(f)
            for line in lines:
                params_dict[line[0]] = line[1]
            f.close()
    else:
        raise ValueError("%s isn't a file!".format(params_file))
    return params_dict


#main method for experimental controller
def main():
    #params_file = "params_edit.csv"
    params_file = "optim_edit.csv"
    
    #init and get rand params for PD and PID-experiments
    params = getParams(params_file)
    #for key, val in params.items():
    #    print("{}:{}".format(key, val))

    #establish timestep and takeoff thresh constants
    TIME_STEP = int(params["QUADCOPTER_TIME_STEP"])
    TAKEOFF_THRESHOLD_VELOCITY = int(params["TAKEOFF_THRESHOLD_VELOCITY"])
    
    #collect waypoints
    waypoints = collectWaypoints()
    #print(waypoints)

    #build robot
    robot = Mavic(waypoints=waypoints,
                  time_step=TIME_STEP,
                  params=params,
                  takeoff_threshold_velocity=TAKEOFF_THRESHOLD_VELOCITY)
    #run robot experiment
    robot.run(params, waypoints)




if __name__ == '__main__':
    main()


