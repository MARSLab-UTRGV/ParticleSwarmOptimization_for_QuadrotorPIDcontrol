"""Flight Trajectory Test for PID Controller.
PID parameters were derived from PSO-tuning
Using similar approach to travel to waypoints and measure MSE
1. Starting Postion-->(0, 0, 0.116)
2. Achieve take-off state to target altitude
3. Hover at target-altitude
4. Once vels==0, change yaw rotation to first waypoint
5. Chg-Yaw towards next target waypoint
6. Change pitch to move towards target
7. Repeat for remaing target waypoints
8. Collect MSE from ref-traj to flight-traj
9. Later, compare with default-controller performance
"""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
###IMPORTS#######################################
from statistics import harmonic_mean
import sys, os
import csv

from numpy import delete
from controller import Supervisor
from cmath import pi

try: 
    import numpy as np
    from csv import DictWriter
    from simple_pid import PID
except ImportError:
    sys.exit("Warning: 'numpy' module not found.")
#################################################


###Object-Class for Mavic Drone
class Mavic(Supervisor):
    # Constants, empirically found.
    K_VERTICAL_THRUST = 68.5  # with this thrust, the drone lifts.
    # Vertical offset where the robot actually targets to stabilize itself.
    K_VERTICAL_OFFSET = 0.6
    K_VERTICAL_P = 3.0        # P constant of the vertical PID.
    K_ROLL_P = 50.0           # P constant of the roll PID.
    K_PITCH_P = 30.0          # P constant of the pitch PID.

    #MAX_YAW_DISTURBANCE = 0.4
    #MAX_YAW_DISTURBANCE = 0.9-best
    MAX_YAW_DISTURBANCE = float(pi)
    MAX_PITCH_DISTURBANCE = -1.0
    
    
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
        self.drone_states = {'A': 'Grounded',
                       'B': 'Takeoff',
                       'C': 'Hover',
                       'D': 'Travel',
                       'E': 'Land'}
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
        self.current_pose = self.gps.getValues()  # X, Y, Z, roll, pitch, yaw

        #including previous-pose to calculate velocity
        self.previous_pose = 6 * [0]
        
        #arrays to sample acceleration to check for:
        # (x, y, z, rol, pit, yaw) steadiness-->vels == 0 
        self.xacc_arr = np.full((1, 20), 99.9)[0]
        self.yacc_arr = np.full((1, 20), 99.9)[0]
        self.altacc_arr = np.full((1, 20), 99.9)[0]
        self.rollacc_arr = np.full((1, 20), 99.9)[0]
        self.pitchacc_arr = np.full((1, 20), 99.9)[0]
        self.yawacc_arr = np.full((1, 20), 99.9)[0]
        self.acc_idx = 0
            
        #init. setpoints for PD-Trajectory Controller
        self.xpos_setpoint = 0.0001#replace with initial waypoint
        self.ypos_setpoint = 0.0001#replace with initial waypoint
        #init PD controllers for xpos and ypos
        self.xposPD = PID(float(params["x_Kp"]), 0, float(params["x_Kd"]), setpoint=float(self.xpos_setpoint))
        self.yposPD = PID(float(params["y_Kp"]), 0, float(params["y_Kd"]), setpoint=float(self.ypos_setpoint))

        #init. setpoints for PID
        self.yaw_setpoint = 0
        self.pitch_setpoint = 0
        self.roll_setpoint = 0
        self.throttle_setpoint = 1
        #init PID controllers for roll, pitch, throttle, yaw
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
                          #output_limits=(-10.0, 10.0),
                          setpoint=float(self.yaw_setpoint))
        
        #target values
        #set target altitude from given z-waypoint
        try:
            #Drone moves to Takeoff state
            self.target_position = waypoints[0]
            print("Using first waypt: {}".format(self.target_position))
            if self.target_position[2] > 0.5:
                self.target_altitude = float(self.target_position[2])
                self.drone_state = self.drone_states['B']
                if self.drone_state == 'Takeoff':
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
        self.waypoint_index = 0
        self.target_alt_thresh = 0.2
        self.pitch_velocity = 0#m/s
        self.pitch_waypoints = [(self.xpos_setpoint, self.ypos_setpoint)]
        self.pitch_waypt_idx = 0
        self.pitch_tgt_waypt = [(0, 0)]
        # Precision b/w tgt-pos and the robot pos in meters
        self.target_precision = 0.1
        print("Mavic Drone initialized...")


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


    #reset waypoint index to 0
    def resetWayptIdx(self):
        self.waypoint_index = 0
        

    #check for hover state: if current 20 samples == 0, for [ xacc, yacc, zacc]
    def isStable(self, stable_arr):
        if self.acc_idx == 20:
            self.acc_idx = 0
        self.xacc_arr[self.acc_idx] = stable_arr[0]
        self.yacc_arr[self.acc_idx] = stable_arr[1]
        self.altacc_arr[self.acc_idx] = stable_arr[2]
        self.rollacc_arr[self.acc_idx] = stable_arr[3]
        self.pitchacc_arr[self.acc_idx] = stable_arr[4]
        self.yawacc_arr[self.acc_idx] = stable_arr[5]
        self.acc_idx += 1
        
        stable_list = np.array([abs(np.mean(self.xacc_arr)) <= 5e-4,
                                abs(np.mean(self.yacc_arr)) <= 5e-4,
                                np.round(abs(np.mean(self.altacc_arr)), 2) == 9.81,
                                abs(np.mean(self.rollacc_arr)) <= 2e-4,
                                abs(np.mean(self.pitchacc_arr)) <= 2e-4,
                                abs(np.mean(self.yacc_arr)) <= 4e-4])
        
        #stable_dict = {'xacc': abs(np.mean(self.xacc_arr)) <= 5e-4,
        #                'yacc': abs(np.mean(self.yacc_arr)) <= 5e-4,
        #                'altacc': np.round(abs(np.mean(self.altacc_arr)), 2) == 9.81,
        #                'rollacc': abs(np.mean(self.rollacc_arr)) <= 2e-4,
        #                'pitchacc': abs(np.mean(self.pitchacc_arr)) <= 2e-4,
        #                'yawacc': abs(np.mean(self.yacc_arr)) <= 4e-4
        #                }
        #for key, val in stable_dict.items():
        #    print("{}: {}".format(key, val))
        return np.all(stable_list)
 

    #return positional velocities of robot
    def getVelocities(self, timestep):
        x_vel = (self.current_pose[0] - self.previous_pose[0]) / timestep
        y_vel = (self.current_pose[1] - self.previous_pose[1]) / timestep
        z_vel = (self.current_pose[2] - self.previous_pose[2]) / timestep
        #print("vels: {:.4f}, {:.4f}, {:.4f}".format(self.current_pose[0], self.previous_pose[0], timestep))
        #print("altitude vel: {:.4f}".format(z_vel))
        return x_vel, y_vel, z_vel


    def move_to_target(self):
        print("heading to tgt-pos: {}".format(self.target_position))
    
    
    #run experiment for analyzing yaw change to waypoint
    def run(self, files_dict, waypoints, params):
        t1 = self.getTime()
        #calcd_time_step = (self.time_step * 4) / 1000
        calcd_time_step = self.time_step / 1000
        print("using calcd time step for PIDs: {}".format(calcd_time_step))

        self.xposPD.sample_time = calcd_time_step
        self.yposPD.sample_time = calcd_time_step
        self.pitchPID.sample_time = calcd_time_step
        self.rollPID.sample_time = calcd_time_step
        self.yawPID.sample_time = calcd_time_step
        self.throttlePID.sample_time = calcd_time_step

        #self.xposPD.proportional_on_measurement = True
        #self.yposPD.proportional_on_measurement = True
        #self.pitchPID.proportional_on_measurement = True
        #self.rollPID.proportional_on_measurement = True
        #self.yawPID.proportional_on_measurement = True
        #self.throttlePID.proportional_on_measurement = True
        print("starting to run experiment...")
        
        #verify tgt-altitude is > 0.5
        takeoff_bool = False
        if self.target_altitude > 0.5:
            takeoff_bool = True
            
        #begin flgt-traj. experiment
        while self.step(self.time_step) != -1 and takeoff_bool:

            #collect current attitude
            roll = self.imu.getRollPitchYaw()[0]# + pi / 2.0
            pitch = self.imu.getRollPitchYaw()[1]
            #yaw = self.compass.getValues()[0]
            yaw = self.imu.getRollPitchYaw()[2]
            #collect  current position
            x_pos, y_pos, altitude = self.gps.getValues()
            roll_acceleration, pitch_acceleration, yaw_acceleration = self.gyro.getValues()
            self.set_position([x_pos, y_pos, altitude, roll, pitch, yaw])

            #collect velocity
            x_vel, y_vel, alt_vel = self.getVelocities(calcd_time_step)
            self.previous_pose = self.current_pose
            
            #collect acceleration
            x_acc, y_acc, alt_acc = self.accelerometer.getValues()

            #write mavic2pro state
            mavic_state = [x_pos, y_pos, altitude, 
                           roll, pitch, yaw,
                           x_vel, y_vel, alt_vel,
                           x_acc, y_acc, alt_acc,
                           roll_acceleration, pitch_acceleration, yaw_acceleration,
                           self.getTime()]
            writeMavicState(mavic_state, files_dict['state-file'])
            #print("rPID: {:.4f} pPID: {:.4f} yPID: {:.4f} tPID: {:.4f}".format(self.rollPID(roll), 
            #                                                      self.pitchPID(pitch), 
            #                                                      self.yawPID(yaw),
            #                                                      self.throttlePID(altitude)))

            #arr used to hold accelerations to determine stability
            stable_arr = np.array([x_acc, y_acc, alt_acc, 
                                   roll_acceleration, pitch_acceleration, yaw_acceleration])
            #flags to enter Hover state from Takeoff State
            takeoff_to_hover_bool = [altitude >= self.target_altitude - self.target_alt_thresh,
                          self.isStable(stable_arr),
                          self.drone_state == 'Takeoff']
            #flags to enter into Travel state from Hover state
            hover_to_travel_bool = [self.drone_state == 'Hover']
            #flags to enter into Hover state from Travel state
            travel_to_hover_bool = [self.drone_state == 'Travel']
            
            
            #if bot is in Takeoff state
            if np.all(takeoff_to_hover_bool):
                self.drone_state = self.drone_states['C']
                print("in Hover state...")
                    
            #if bot is in Hover state
            if np.all(hover_to_travel_bool):
                #if not at end of waypoints, make tgt-pos next waypoint
                if self.waypoint_index < len(waypoints):
                    self.waypoint_index += 1
                    self.target_position = waypoints[self.waypoint_index]
                    self.drone_state = self.drone_states['D']
                    print("in Travel state...")
                    
            #if bot is in Travel state
            if np.all(travel_to_hover_bool):
                #if bot is at tgt-pos w/in a target_precision->move into hover state at tgt-pos
                if all([abs(x1 - x2) < self.target_precision for (x1, x2) in zip(self.target_position[0:2], self.current_pose[0:2])]):
                    self.drone_state = self.drone_states['C']
                    print("Tgt-Pos reached, drone in Hover state...")
                #else->have bot keep moving to tgt-pos
                else:
                    self.move_to_target()
                
                
                
            
            
            
            
            #collect inputs
            #roll
            roll_input = float(params["k_roll_p"]) * clamp(roll, -1, 1) + roll_acceleration + self.rollPID(roll) + clamp(self.yposPD(y_pos), -1, 1)
            #pitch
            pitch_input = float(params["k_pitch_p"]) * clamp(pitch, -1, 1) + pitch_acceleration + self.pitchPID(pitch) + clamp(self.xposPD(-x_pos), -1, 1)
            #yaw
            yaw_input = self.yawPID(yaw)
            #thrust
            #print("climbing to target altitude: {:.4f}".format(float(self.target_altitude)))
            diff_altitude = self.target_altitude - altitude + float(params["k_vertical_offset"])
            #print("diff_alt: {:.4f}, {}".format(diff_altitude, type(diff_altitude)))
            clamped_difference_altitude = clamp(self.target_altitude - altitude + float(params["k_vertical_offset"]), -1.0, 1.0)
            #print("clamp_diff: {:.4f}, {}".format(clamped_difference_altitude, type(clamped_difference_altitude)))
            #vertical_input = self.throttlePID(clamped_difference_altitude )
            vertical_input = self.throttlePID(altitude)
            #print("inputs-->roll: {:.4f}, pitch: {:.4f}, yaw: {:.4f}, vertical: {:.4f}".format(roll_input, pitch_input, yaw_input, vertical_input))

            #write PID and rotational and vertical inputs
            pid_inputs = [clamp(self.xposPD(-x_pos), -1, 1), clamp(self.yposPD(y_pos), -1, 1), 
                          self.rollPID(roll), self.pitchPID(pitch), 
                          self.yawPID(yaw), self.throttlePID(clamped_difference_altitude),
                          roll_input, pitch_input, 
                          yaw_input, vertical_input,
                          diff_altitude, clamped_difference_altitude]
            writePIDandInputs(pid_inputs, files_dict['pid-file'])
            
            ##Pause simulation when conditions are met
            #set exit-condition for time
            time_bool = False
            if self.getTime() > float(60):
                time_bool = True
                
            #set exit-conditions for position
            #--->2 conditions: if bot is in hover->stay at curr_waypt
            #----------------->if bot is in pitch-chg state->swap pos cond with angle_tol
            wypnt_tolerance = float(10)
            position_cond = np.array([x_pos, y_pos, altitude])
            position_bool = False
            #only checking for going above target altitude
            if (position_cond[2] > self.target_altitude + (wypnt_tolerance * 5)):#for altitude
                position_bool = True
                print("z became true")
            #if drone is in Pitch-Chg state-->get current pitch waypt-tgt(x, y)
            #if self.drone_state == "Travel":
            #    pitch_waypt = self.pitch_waypoints[self.pitch_waypt_idx]
            #else-->drone is not in Pitch-Chg state-->use current x-setpoint, y-setpoint
            #else:
            #    pitch_waypt = (self.xpos_setpoint ,self.ypos_setpoint)
            pitch_waypt = (self.xpos_setpoint ,self.ypos_setpoint)
            #check x and y position cond
            if position_cond[0] > pitch_waypt[0] + wypnt_tolerance or position_cond[0] < pitch_waypt[0] - wypnt_tolerance:
                position_bool = True
                print("x became true")
            if position_cond[1] > pitch_waypt[1] + wypnt_tolerance or position_cond[1] < pitch_waypt[1] - wypnt_tolerance:
                position_bool = True
                print("y became true")

            #set exit-conditions for attitude
            attitude_cond_r_p = np.array([roll, pitch])#roll and pitch cond
            attitude_cond_y = np.array([yaw])#yaw cond
            attitude_bool = False
            #only checking for roll and pitch since we change yaw
            for att in attitude_cond_r_p:
                if (att > 1.25) or (att < -1.25):
                    attitude_bool = True
                    print("pitch too much...")

            #set exit-conditions for velocity
            alt_bool = False
            if alt_vel < -float(8):
                alt_bool = True

            #motor inputs
            front_left_motor_input = float(params["k_vertical_thrust"]) + vertical_input - yaw_input + pitch_input - roll_input
            front_right_motor_input = float(params["k_vertical_thrust"]) + vertical_input + yaw_input + pitch_input + roll_input
            rear_left_motor_input = float(params["k_vertical_thrust"]) + vertical_input + yaw_input - pitch_input - roll_input
            rear_right_motor_input = float(params["k_vertical_thrust"]) + vertical_input - yaw_input - pitch_input + roll_input
            #print("motors:\n{:.4f}, {:.4f}, {:.4f}, {:.4f}".format(front_left_motor_input, front_right_motor_input, rear_left_motor_input, rear_right_motor_input))

            #set exit-conditions for motor inputs
            mtr_input_cond = np.array([front_left_motor_input, 
                                        front_right_motor_input, 
                                        rear_left_motor_input,
                                        rear_right_motor_input])
            
            mtr_inpt_bool = False
            for inpt in mtr_input_cond:
                if inpt > 300:
                    mtr_inpt_bool = True

            if attitude_bool or mtr_inpt_bool or position_bool or time_bool or alt_bool:
                self.simulationSetMode(0)

            self.front_left_motor.setVelocity(front_left_motor_input)
            self.front_right_motor.setVelocity(-front_right_motor_input)
            self.rear_left_motor.setVelocity(-rear_left_motor_input)
            self.rear_right_motor.setVelocity(rear_right_motor_input)




#clamp-fcn: return min-val b/w input max value
#and (the max b/w curr val and min val)
def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)


#clear file, if exists
def clearFileIfExists(filename):
    if os.path.isfile(filename):
        os.remove(filename)
        

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


#write state of Drone every timestep
def writeMavicState(input_arr, filename):

    field_names = ['x_pos', 'y_pos', 'z_pos', 
                   'roll_rot', 'pitch_rot', 'yaw_rot', 
                   'x_vel', 'y_vel', 'alt_vel',
                   'x_acc', 'y_acc', 'alt_acc',
                   'roll_acc', 'pitch_acc', 'yaw_acc',
                   'timestep']

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
                'timestep': input_arr[15]}

    with open(filename, 'a') as f_obj:
        dictWriter_obj = DictWriter(f_obj, fieldnames=field_names)
        dictWriter_obj.writerow(csv_dict)
        f_obj.close()
        

#Document PID and Inputs per timestep
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


def main():
    #clear output files if they exit
    filedir = r"C:\Users\ericx\OneDrive\Desktop\mine\grad_school\MARS_Lab_Swarm\Gazebo_dragonFly\webots_tutorial\hovering_test_project\python_utilities\PID_VS_Default_traj"
    ###experimental-designator filename
    state_filename = filedir + r"\PIDmavic_state.csv"
    inputs_filename = filedir + r"\PIDinputs.csv"
    clearFileIfExists(state_filename)
    clearFileIfExists(inputs_filename)
    files_dict = {'state-file': state_filename, 'pid-file': inputs_filename}
    
    #collect parameters for PID-experiment
    print("numpy version: {}".format(np.__version__))
    params = dict()
    #read in parameters from file
    with open("params_edit.csv", "r") as f:
        lines = csv.reader(f)
        for line in lines:
            #print(line)
            params[line[0]] = line[1]
    #for key, val in params.items():
    #    print("{}:{}".format(key, val))
        
    #collect waypoints
    waypoints = collectWaypoints()
    
    #time-step=32ms
    TIME_STEP = int(params["QUADCOPTER_TIME_STEP"])
    #takeoff-thresh-vel = 160
    TAKEOFF_THRESHOLD_VELOCITY = int(params["TAKEOFF_THRESHOLD_VELOCITY"])
    
    #build robot
    robot = Mavic(waypoints=waypoints,
                  time_step=TIME_STEP,
                  params=params,
                  takeoff_threshold_velocity=TAKEOFF_THRESHOLD_VELOCITY)
    #run robot experiment
    robot.run(files_dict, waypoints, params=params)
        
    
        
    
if __name__ == '__main__':
    main()