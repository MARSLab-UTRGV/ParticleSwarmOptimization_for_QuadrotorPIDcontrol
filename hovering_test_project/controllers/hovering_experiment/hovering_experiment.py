"""hovering_experiment controller.
1. Starting Position-->(-8, 0, 0.15)
2. Run World for 4s then move drone to (-8, 0, 1)
3. Hover for 10s
4. then move to (8, 0, 1)
5. hover for 4s then land 
6. close experiment

Result Exploration: 
1. find how drone moves from start to hover
2. find how drone moves from hovering pos to waypoint pos
3. find how drone hovers at waypoint pos
3a. look at orientation when reaches waypoint
4. look how fast drone lands???
"""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
###IMPORTS#######################################
from genericpath import isfile
from msilib.schema import File
import sys, os
import csv
from pickle import DICT
from tkinter import VERTICAL, Y
from controller import Robot
from controller import Supervisor
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


def writeMavicState(input_arr, filename):
    #filename = r"C:\Users\ericx\OneDrive\Desktop\mine\grad_school\MARS_Lab_Swarm\Gazebo_dragonFly\webots_tutorial\hovering_test_project\python_utilities"
    ###experimental-designator filename
    #filename += r"\mavic_state.csv"

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
    #filename = r"C:\Users\ericx\OneDrive\Desktop\mine\grad_school\MARS_Lab_Swarm\Gazebo_dragonFly\webots_tutorial\hovering_test_project\python_utilities"
    ###experimental-designator filename
    #filename += r"\PID_and_inputs.csv"

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
    # Precision between the target position and the robot position in meters
    target_precision = 0.1

    def __init__(self, time_step, params, takeoff_threshold_velocity=1):
        Supervisor.__init__(self)

        try:
            self.time_step = time_step
        except:
            self.time_step = int(self.getBasicTimeStep())
        print("using timestep: {}".format(self.time_step))

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
        self.xposPD = PID(float(params["x_Kp"]), 0, float(params["x_Kd"]), setpoint=float(xpos_setpoint))
        self.yposPD = PID(float(params["y_Kp"]), 0, float(params["y_Kd"]), setpoint=float(ypos_setpoint))

        #init. setpoints for PID
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
                          #output_limits=(-10.0, 10.0),
                          setpoint=float(yaw_setpoint))


        #enable current pose for robot
        #self.current_pose = 6 * [0]  # X, Y, Z, yaw, pitch, roll
        self.current_pose = self.gps.getValues()  # X, Y, Z, yaw, pitch, roll
        #including previous-pose to calculate velocity
        self.previous_pose = 6 * [0]
        #target values
        self.target_position = [0, 0, 0] #X, Y, Yaw
        self.target_index = 0
        self.target_altitude = float(params["target_altitude"])
        self.throttlePID.setpoint = self.target_altitude###setting trottlePID setpoint to params)
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
        print("vels: {:.4f}, {:.4f}, {:.4f}".format(self.current_pose[0], self.previous_pose[0], timestep))
        print("altitude vel: {:.4f}".format(z_vel))
        return x_vel, y_vel, z_vel

    
    def run(self, files_dict, waypoints, params):
        t1 = self.getTime()
        #print("using calcd time step for PIDs: {}".format((self.time_step * 4) / 1000))
        #calcd_time_step = (self.time_step * 4) / 1000
        print("using calcd time step for PIDs: {}".format(self.time_step / 1000))
        calcd_time_step = self.time_step / 1000

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

        roll_disturbance = 0
        pitch_disturbance = 0
        yaw_disturbance = 0
        print("starting to run experiment...")

        # Specify the patrol coordinates or waypoints

        while self.step(self.time_step) != -1:

            #roll, pitch, yaw = self.imu.getRollPitchYaw()
            roll = self.imu.getRollPitchYaw()[0]# + pi / 2.0
            pitch = self.imu.getRollPitchYaw()[1]
            #yaw = self.compass.getValues()[0]
            yaw = self.imu.getRollPitchYaw()[2]

            x_pos, y_pos, altitude = self.gps.getValues()
            roll_acceleration, pitch_acceleration, yaw_acceleration = self.gyro.getValues()
            self.set_position([x_pos, y_pos, altitude, roll, pitch, yaw])

            #collect velocity
            x_vel, y_vel, alt_vel = self.getVelocities(calcd_time_step)
            self.previous_pose = self.current_pose


            #write mavic2pro state
            mavic_state = [x_pos, y_pos, altitude, roll, pitch, yaw,
                           x_vel, y_vel, alt_vel,
                           roll_acceleration, pitch_acceleration, yaw_acceleration,
                           self.getTime()]
            writeMavicState(mavic_state, files_dict['state-file'])

            print("rPID: {:.4f} pPID: {:.4f} yPID: {:.4f} tPID: {:.4f}".format(self.rollPID(roll), 
                                                                  self.pitchPID(pitch), 
                                                                  self.yawPID(yaw),
                                                                  self.throttlePID(altitude)))

            #collect inputs
            roll_input = float(params["k_roll_p"]) * clamp(roll, -1, 1) + roll_acceleration + self.rollPID(roll) + self.yposPD(y_pos)
            #roll_input = float(params["k_roll_p"]) * clamp(roll, -1, 1) + roll_acceleration + self.rollPID(roll)
            #roll_input = float(params["k_roll_p"]) * roll + self.rollPID(roll)
            #roll_input = self.xposPD(x_pos) * clamp(roll, -1, 1) + roll_acceleration + self.rollPID(roll)
            #roll_input = self.rollPID(roll)# + roll_acceleration# + self.rollPID(roll)
            #roll_input = self.xposPD(x_pos) + float(params["k_roll_p"]) * clamp(roll, -1, 1) + roll_acceleration + self.rollPID(roll)
            #roll_input = float(params["k_roll_p"]) * clamp(roll, -1, 1) + roll_acceleration + self.rollPID(roll)

            pitch_input = float(params["k_pitch_p"]) * clamp(pitch, -1, 1) + pitch_acceleration + self.pitchPID(pitch) + self.xposPD(-x_pos)
            #pitch_input = float(params["k_pitch_p"]) * clamp(pitch, -1, 1) + pitch_acceleration + self.pitchPID(pitch)
            #pitch_input = float(params["k_pitch_p"]) * pitch + self.pitchPID(pitch)
            #pitch_input = self.yposPD(y_pos) * clamp(pitch, -1, 1) + pitch_acceleration + self.pitchPID(pitch)
            #pitch_input = self.pitchPID(pitch)# + pitch_acceleration# + self.pitchPID(pitch)
            #pitch_input = self.yposPD(y_pos) + float(params["k_pitch_p"]) * clamp(pitch, -1, 1) + pitch_acceleration + self.pitchPID(pitch)
            #pitch_input = float(params["k_pitch_p"]) * clamp(pitch, -1, 1) + pitch_acceleration + self.pitchPID(pitch)
            
            print("climbing to target altitude: {:.4f}".format(float(self.target_altitude)))
            diff_altitude = self.target_altitude - altitude + float(params["k_vertical_offset"])
            print("diff_alt: {:.4f}, {}".format(diff_altitude, type(diff_altitude)))
            clamped_difference_altitude = clamp(self.target_altitude - altitude + float(params["k_vertical_offset"]), -1.0, 1.0)
            #clamped_difference_altitude = float(self.target_altitude) - altitude + float(params["k_vertical_offset"])
            #vertical_input = self.throttlePID(altitude) * pow(clamped_difference_altitude, 3)
            print("clamp_diff: {:.4f}, {}".format(clamped_difference_altitude, type(clamped_difference_altitude)))
            #vertical_input = self.throttlePID(clamped_difference_altitude )
            vertical_input = self.throttlePID(altitude)
            #vertical_input = self.K_VERTICAL_P * pow(clamped_difference_altitude, 3.0)
            #vertical_input = self.throttlePID(altitude)
            print("vert input: {}".format(vertical_input))
            
            #yaw_input = clamp(yaw, -1, 1) + yaw_acceleration + self.yawPID(yaw)
            yaw_input = self.yawPID(yaw)
            
            print("inputs-->roll: {:.4f}, pitch: {:.4f}, yaw: {:.4f}, vertical: {:.4f}".format(roll_input, pitch_input, yaw_input, vertical_input))
            #roll_input = self.K_ROLL_P * clamp(roll, -1, 1) + roll_acceleration + roll_disturbance
            #pitch_input = self.K_PITCH_P * clamp(pitch, -1, 1) + pitch_acceleration + pitch_disturbance
            #yaw_input = yaw_disturbance
            #clamped_difference_altitude = clamp(self.target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1)

            #write PID and rotational and vertical inputs
            pid_inputs = [self.xposPD(-x_pos), self.yposPD(y_pos), 
                          self.rollPID(roll), self.pitchPID(pitch), 
                          self.yawPID(yaw), self.throttlePID(clamped_difference_altitude),
                          roll_input, pitch_input, 
                          yaw_input, vertical_input,
                          diff_altitude, clamped_difference_altitude]
            writePIDandInputs(pid_inputs, files_dict['pid-file'])


            ##Pause simulation when conditions are met
            #set exit-condition for time
            time_bool = False
            if self.getTime() > float(30):
                time_bool = True

            #set exit-conditions for position
            wypnt_tolerance = float(2)
            position_cond = np.array([x_pos, y_pos, altitude])
            position_bool = False
            if (position_cond[0] > waypoints[0] + wypnt_tolerance) or (position_cond[0] < waypoints[0] - wypnt_tolerance):
                position_bool = True
                print("x became true")
            elif (position_cond[1] > waypoints[1] + wypnt_tolerance) or (position_cond[1] < waypoints[1] - wypnt_tolerance):
                position_bool = True
                print("y became true")
            elif (position_cond[2] > waypoints[2] + (wypnt_tolerance * 5)):#for altitude
                position_bool = True
                print("z became true")

            #set exit-conditions for attitude
            attitude_cond_r_p = np.array([roll, pitch])#roll and pitch cond
            attitude_cond_y = np.array([yaw])#yaw cond
            attitude_bool = False
            for att in attitude_cond_r_p:
                if (att > 1.25) or (att < -1.25):
                    attitude_bool = True
            if attitude_cond_y > float(pi) or attitude_cond_y < -float(pi):
                attitude_bool = True

            #set exit-conditions for velocity
            alt_bool = False
            if alt_vel < -float(8):
                alt_bool = True

            #motor inputs
            front_left_motor_input = float(params["k_vertical_thrust"]) + vertical_input - yaw_input + pitch_input - roll_input
            front_right_motor_input = float(params["k_vertical_thrust"]) + vertical_input + yaw_input + pitch_input + roll_input
            rear_left_motor_input = float(params["k_vertical_thrust"]) + vertical_input + yaw_input - pitch_input - roll_input
            rear_right_motor_input = float(params["k_vertical_thrust"]) + vertical_input - yaw_input - pitch_input + roll_input

            #front_left_motor_input = float(params["k_vertical_thrust"]) + vertical_input - roll_input - pitch_input + yaw_input
            #front_right_motor_input = float(params["k_vertical_thrust"]) + vertical_input + roll_input - pitch_input - yaw_input
            #rear_left_motor_input = float(params["k_vertical_thrust"]) + vertical_input - roll_input + pitch_input - yaw_input
            #rear_right_motor_input = float(params["k_vertical_thrust"]) + vertical_input + roll_input + pitch_input + yaw_input
            print("motors:\n{:.4f}, {:.4f}, {:.4f}, {:.4f}".format(front_left_motor_input, front_right_motor_input, rear_left_motor_input, rear_right_motor_input))

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




#main function
def main():
    #clear output files if they exit
    filedir = r"C:\Users\ericx\OneDrive\Desktop\mine\grad_school\MARS_Lab_Swarm\Gazebo_dragonFly\webots_tutorial\hovering_test_project\python_utilities"
    ###experimental-designator filename
    state_filename = filedir + r"\mavic_state.csv"
    pid_filename = filedir + r"\PID_and_inputs.csv"
    clearFileIfExists(state_filename)
    clearFileIfExists(pid_filename)
    files_dict = {'state-file': state_filename, 'pid-file': pid_filename}

    
    #collect parameters for PID-experiment
    print("numpy version: {}".format(np.__version__))
    params = dict()

    with open("params_edit.csv", "r") as f:
        lines = csv.reader(f)
        for line in lines:
            print(line)
            params[line[0]] = line[1]
    for key, val in params.items():
        print("{}:{}".format(key, val))

    #collect waypoints
    waypoints = np.array([0.0001, 0.0001, float(params["target_altitude"])])
            
    TIME_STEP = int(params["QUADCOPTER_TIME_STEP"])
    TAKEOFF_THRESHOLD_VELOCITY = int(params["TAKEOFF_THRESHOLD_VELOCITY"])
    robot = Mavic(time_step=TIME_STEP,
                  params=params,
                  takeoff_threshold_velocity=TAKEOFF_THRESHOLD_VELOCITY)
    robot.run(files_dict, waypoints, params=params)


if __name__ == '__main__':
    main()





