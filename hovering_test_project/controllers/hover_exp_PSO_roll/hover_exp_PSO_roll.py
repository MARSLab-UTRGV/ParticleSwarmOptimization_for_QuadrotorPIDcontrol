"""hover_exp_PSO_roll controller.
1. Start PSO for Roll PID-params
2. Params Domain: 
    {'P': [1, 5, steps of 0.2]}
    {'I': [0.1]} #not yet tuned
    {'D': [1, 5, steps of 0.2]}
3. PSO:
    pbest_obj-fcn = (ln(pos_ovrshot/pos_expect +1) + ln(time_ovrshot/time_expect +1))
    X = {PID parameters} --(maybe try to input tgt-altitude)
    V = initial set to same-dims as X, all 0s
    V_update = wgt*V + c1*r1*(pbest -X) + c2*r2*(gbest -X)
"""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
###IMPORTS#######################################
from genericpath import isfile
import sys, os, time
import csv
import struct
from pickle import DICT
from tkinter import VERTICAL
from controller import Robot, Emitter, Receiver
from cmath import pi
try: 
    import numpy as np
    from csv import DictWriter
    from simple_pid import PID
except ImportError:
    sys.exit("Warning: 'numpy' module not found.")
#################################################

#clamp-fcn: return min-val b/w input max value
#and (the max b/w curr val and min val)
def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)

def clearFileIfExists(filename):
    if os.path.isfile(filename):
        os.remove(filename)

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

#collect waypoints from logfile
def collectWaypoints() -> list:
    waypoint_list = []
    filename = r"waypoint_logfile.txt"
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            points = line.strip().split(" ")
            waypoint_list.append([float(points[0]), float(points[1])])
    return waypoint_list

def writeMavicState(input_arr, filename):
    field_names = ['x_pos', 'y_pos', 'z_pos', 
                   'roll_rot', 'pitch_rot', 'yaw_rot', 
                   'x_vel', 'y_vel', 'alt_vel',
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
                'roll_acc': input_arr[9],
                'pitch_acc': input_arr[10],
                'yaw_acc': input_arr[11],
                'timestep': input_arr[12]}

    with open(filename, 'a') as f_obj:
        dictWriter_obj = DictWriter(f_obj, fieldnames=field_names)
        dictWriter_obj.writerow(csv_dict)
        f_obj.close()

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

###Object-Class for Mavic Drone
class Mavic(Robot):

    def __init__(self, time_step, params, takeoff_threshold_velocity=1):
        Robot.__init__(self)

        try:
            self.time_step = time_step
        except:
            self.time_step = int(self.getBasicTimeStep())
        print("mavic using timestep: {}".format(self.time_step))

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

        #init. setpoints for PD-Trajectory Controller
        xpos_setpoint = 0.0001#replace with initial waypoint
        ypos_setpoint = 0.0001#replace with initial waypoint
        #init PD controllers for xpos and ypos
        self.xposPD = PID(float(params["x_Kp"]), 
                          0, 
                          float(params["x_Kd"]), 
                          setpoint=float(xpos_setpoint))
        self.yposPD = PID(float(params["y_Kp"]), 
                          0, 
                          float(params["y_Kd"]), 
                          setpoint=float(ypos_setpoint))
        
        #init setpoints for PID-Attitude Controller
        yaw_setpoint = 0
        pitch_setpoint = 0
        roll_setpoint = 0
        throttle_setpoint = 1
        #init PID controllers for roll, pitch, throttle, yaw
        self.pitchPID = PID(float(params["pitch_Kp"]), 
                            float(params["pitch_Ki"]), 
                            float(params["pitch_Kd"]), 
                            setpoint=float(pitch_setpoint))
        self.rollPID = PID(float(params["roll_Kp"]), 
                           float(params["roll_Ki"]), 
                           float(params["roll_Kd"]), 
                           setpoint=float(roll_setpoint))
        self.throttlePID = PID(float(params["throttle_Kp"]), 
                               float(params["throttle_Ki"]), 
                               float(params["throttle_Kd"]), 
                               output_limits=(-50.0, 50.0), 
                               setpoint=float(throttle_setpoint))
        self.yawPID = PID(float(params["yaw_Kp"]), 
                          float(params["yaw_Ki"]), 
                          float(params["yaw_Kd"]), 
                          setpoint=float(yaw_setpoint))

        #enable current pose for robot
        self.current_pose = self.gps.getValues()  # X, Y, Z, yaw, pitch, roll
        #including previous-pose to calculate velocity
        self.previous_pose = 6 * [0]
        #target values
        self.target_position = [0, 0, 0] #X, Y, Yaw
        self.target_index = 0
        self.target_altitude = float(params["target_altitude"])
        self.throttlePID.setpoint = self.target_altitude###setting trottlePID setpoint to params
        print("Mavic Drone initialized...")


    def set_position(self, pos):
        """
        Set the new absolute position of the robot
        Parameters:
            pos (list): [X,Y,Z,yaw,pitch,roll] current absolute position and angles
        """
        self.current_pose = pos

    def set_previous_position(self, pos):
        self.previous_pose = pos

    def getVelocities(self, timestep):
        x_vel = (self.current_pose[0] - self.previous_pose[0]) / timestep
        y_vel = (self.current_pose[1] - self.previous_pose[1]) / timestep
        z_vel = (self.current_pose[2] - self.previous_pose[2]) / timestep
        #print("vels: {:.4f}, {:.4f}, {:.4f}".format(self.current_pose[0], self.previous_pose[0], timestep))
        #print("altitude vel: {:.4f}".format(z_vel))
        return x_vel, y_vel, z_vel

    def run(self, params):
        t0 = self.getTime()
        #print("using calcd time step for PIDs : {}".format((self.time_step * 4) / 1000))
        #calcd_time_step = (self.time_step * 4) / 1000
        print("using calcd time step for PIDs : {}".format((self.time_step) / 1000))
        calcd_time_step = self.time_step / 1000

        self.xposPD.sample_time = calcd_time_step
        self.yposPD.sample_time = calcd_time_step
        self.pitchPID.sample_time = calcd_time_step
        self.rollPID.sample_time = calcd_time_step
        self.yawPID.sample_time = calcd_time_step
        self.throttlePID.sample_time = calcd_time_step

        roll_disturbance = 0
        pitch_disturbance = 0
        yaw_disturbance = 0
        print("starting to run PSO experiment...")

        # Specify the patrol coordinates
        waypoints = collectWaypoints()
        #print(waypoints)

        print("setting target altitude: {}m".format(self.target_altitude))
        while self.step(self.time_step) != -1:
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

            #collect velocity
            x_vel, y_vel, alt_vel = self.getVelocities(calcd_time_step)
            self.previous_pose = self.current_pose
            vel_msg = struct.pack("ddd", x_vel, y_vel, alt_vel)
            self.mvc_emtr.send(vel_msg)

            #acceleration_msg
            roll_acceleration, pitch_acceleration, yaw_acceleration = self.gyro.getValues()
            acc_msg = struct.pack("ddd", roll_acceleration, pitch_acceleration, yaw_acceleration)
            self.mvc_emtr.send(acc_msg)

            #time_msg
            time_msg = struct.pack("d", self.getTime())
            self.mvc_emtr.send(time_msg)

            #collect inputs
            #roll_input = self.xposPD(x_pos) + float(params["k_roll_p"]) * clamp(roll, -1, 1) + roll_acceleration + self.rollPID(roll)
            #roll_input = float(params["k_roll_p"]) * clamp(roll, -1, 1) + roll_acceleration + self.rollPID(roll)
            roll_input = float(params["k_roll_p"]) * clamp(roll, -1, 1) + roll_acceleration + self.rollPID(roll) + self.yposPD(y_pos)
            
            #pitch_input = self.yposPD(y_pos) + float(params["k_pitch_p"]) * clamp(pitch, -1, 1) + pitch_acceleration + self.pitchPID(pitch)
            #pitch_input = float(params["k_pitch_p"]) * clamp(pitch, -1, 1) + pitch_acceleration + self.pitchPID(pitch)
            pitch_input = float(params["k_pitch_p"]) * clamp(pitch, -1, 1) + pitch_acceleration + self.pitchPID(pitch) + self.xposPD(-x_pos)
            
            #print("climbing to target altitude: {:.4f}".format(float(self.target_altitude)))
            diff_altitude = self.target_altitude - altitude + float(params["k_vertical_offset"])
            clamped_difference_altitude = clamp(diff_altitude, -1.0, 1.0)
            vertical_input = self.throttlePID(altitude)
            
            yaw_input = self.yawPID(yaw)
            #yaw_input = yaw + yaw_acceleration + self.yawPID(yaw)
            #print("inputs-->roll: {:.4f}, pitch: {:.4f}, yaw: {:.4f}, vertical: {:.4f}".format(roll_input, pitch_input, yaw_input, vertical_input))

            #print("xPD: {:.4f} yPD: {:.4f}".format(self.xposPD(x_pos), self.yposPD(y_pos)))

            #print("rPID: {:.4f} pPID: {:.4f} yPID: {:.4f} tPID: {:.4f}".format(self.rollPID(roll), 
            #                                                      self.pitchPID(pitch), 
            #                                                      self.yawPID(yaw),
            #                                                      self.throttlePID(clamped_difference_altitude)))
           

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
            posPID_msg = struct.pack("dd", self.xposPD(x_pos), self.yposPD(y_pos))
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

            self.front_left_motor.setVelocity(front_left_motor_input)
            self.front_right_motor.setVelocity(-front_right_motor_input)
            self.rear_left_motor.setVelocity(-rear_left_motor_input)
            self.rear_right_motor.setVelocity(rear_right_motor_input)



#main function
def main():
    #params_file = "params_edit.csv"
    params_file = "optim_edit.csv"

    #init and get rand params for PD and PID-experiments
    params = getParams(params_file)
    #for key, val in params.items():
    #    print("{}:{}".format(key, val))
        
    TIME_STEP = int(params["QUADCOPTER_TIME_STEP"])
    TAKEOFF_THRESHOLD_VELOCITY = int(params["TAKEOFF_THRESHOLD_VELOCITY"])
    
    # create the Robot instance.
    robot = Mavic(time_step=TIME_STEP,
                  params=params,
                  takeoff_threshold_velocity=TAKEOFF_THRESHOLD_VELOCITY)
    robot.run(params=params)

if __name__ == '__main__':
    main()