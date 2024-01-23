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
from tkinter import VERTICAL, W, Y
from controller import Robot
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
                   'roll_acc', 'pitch_acc', 'yaw_acc',
                   'timestep']

    csv_dict = {'x_pos': input_arr[0],
                'y_pos': input_arr[1],
                'z_pos': input_arr[2],
                'roll_rot': input_arr[3],
                'pitch_rot': input_arr[4],
                'yaw_rot': input_arr[5],
                'roll_acc': input_arr[6],
                'pitch_acc': input_arr[7],
                'yaw_acc': input_arr[8],
                'timestep': input_arr[9]}

    with open(filename, 'a') as f_obj:
        dictWriter_obj = DictWriter(f_obj, fieldnames=field_names)
        dictWriter_obj.writerow(csv_dict)
        f_obj.close()

def writePIDandInputs(input_arr, filename):
    #filename = r"C:\Users\ericx\OneDrive\Desktop\mine\grad_school\MARS_Lab_Swarm\Gazebo_dragonFly\webots_tutorial\hovering_test_project\python_utilities"
    ###experimental-designator filename
    #filename += r"\PID_and_inputs.csv"

    field_names = ['rollPID', 'pitchPID', 'yawPID', 'throttlePID',
                   'roll_input', 'pitch_input', 'vertical_input', 'yaw_input', 
                   'diff_altitude', 'clampd_diff_altitude']

    csv_dict = {'rollPID': input_arr[0],
                'pitchPID': input_arr[1],
                'yawPID': input_arr[2],
                'throttlePID':input_arr[3],
                'roll_input': input_arr[4],
                'pitch_input': input_arr[5],
                'vertical_input': input_arr[6],
                'yaw_input': input_arr[7],
                'diff_altitude': input_arr[8],
                'clampd_diff_altitude': input_arr[9]}

    with open(filename, 'a') as f_obj:
        dictWriter_obj = DictWriter(f_obj, fieldnames=field_names)
        dictWriter_obj.writerow(csv_dict)
        f_obj.close()

###Object-Class for Mavic Drone
class Mavic(Robot):
    # Constants, empirically found.
    K_VERTICAL_THRUST = 68.5  # with this thrust, the drone lifts.
    # Vertical offset where the robot actually targets to stabilize itself.
    K_VERTICAL_OFFSET = 0.6
    K_VERTICAL_P = 3.0        # P constant of the vertical PID.
    K_ROLL_P = 50.0           # P constant of the roll PID.
    K_PITCH_P = 50.0          # P constant of the pitch PID.

    #MAX_YAW_DISTURBANCE = 0.4
    MAX_YAW_DISTURBANCE = 1.0#best
    #MAX_YAW_DISTURBANCE = float(pi)
    MAX_PITCH_DISTURBANCE = -1.0
    # Precision between the target position and the robot position in meters
    target_precision = 0.2

    def __init__(self, time_step, params, takeoff_threshold_velocity=1):
        Robot.__init__(self)

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
            

        #init. setpoints for PID
        yaw_setpoint = -1
        pitch_setpoint = 0
        roll_setpoint = 0
        throttle_setpoint = 1
        #init PID controllers for roll, pitch, throttle, yaw
        self.pitchPID = PID(float(params["pitch_Kp"]), float(params["pitch_Ki"]), float(params["pitch_Kd"]), setpoint=float(pitch_setpoint))
        self.rollPID = PID(float(params["roll_Kp"]), float(params["roll_Ki"]), float(params["roll_Kd"]), setpoint=float(roll_setpoint))
        self.throttlePID = PID(float(params["throttle_Kp"]), float(params["throttle_Ki"]), float(params["throttle_Kd"]), setpoint=throttle_setpoint)
        self.yawPID = PID(float(params["yaw_Kp"]), float(params["yaw_Ki"]), float(params["yaw_Kd"]), setpoint=float(yaw_setpoint))


        #enable current pose for robot
        self.current_pose = 6 * [0]  # X, Y, Z, yaw, pitch, roll
        #taget values
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

    def move_to_target(self, waypoints):
        """
        Move the robot to the given coordinates
        Parameters:
            waypoints (list): list of X,Y coordinates

        Returns:
            yaw_disturbance (float): yaw disturbance (negative value to go on the right)
            //not implemented yet---pitch_disturbance (float): pitch disturbance (negative value to go forward)
        """
        print("changing yaw to tgt-waypoint: {}".format(waypoints))
        if self.target_position[0:2] == [0, 0]: #Initilizing target waypoint
            self.target_position[0:2] = waypoints[0]
        print("tgt-waypoint set to {}".format(self.target_position[0:2]))

        # if the robot is at the position with a precision of target_precision
        if all([abs(x1 - x2) < self.target_precision for (x1, x2) in zip(self.target_position, self.current_pose[0:2])]):
            self.target_index += 1
            #if we are at end of waypoint list-->start from beginning
            if self.target_index > len(waypoints) - 1:
                self.target_index = 0
            self.target_position[0:2] = waypoints[self.target_index]

        #change drone-yaw towards tgt-waypoint
        # This will be in ]-pi;pi]
        self.target_position[2] = np.arctan2(self.target_position[1] - self.current_pose[1], self.target_position[0] - self.current_pose[0])
        print("target-angle: {:.4f}".format(self.target_position[2]))

        # This is now in ]-2pi;2pi[
        angle_left = self.target_position[2] - self.current_pose[5]
        print("angle-left not-nrmlizd: {:.4f}".format(angle_left))

        # Normalize turn angle to ]-pi;pi]
        angle_left = (angle_left + 2 * np.pi) % (2 * np.pi)
        print("norm angle-left: {:.4f}".format(angle_left))

        if (angle_left > np.pi):
            angle_left -= 2 * np.pi
        print("angle-left final: {:.4f}".format(angle_left))

        # Turn the robot to the left or to the right according the value and the sign of angle_left
        yaw_disturbance = self.MAX_YAW_DISTURBANCE * angle_left / (2 * np.pi)
        print("yaw-disturbs: {:.4f}".format(yaw_disturbance))
        print("just angle: {:.4f}".format(angle_left / (2 * np.pi)))

        return clamp(angle_left / (2 * np.pi), -1 ,1)
        #return yaw_disturbance



    def run(self, files_dict, params):
        t1 = self.getTime()
        print("using time step: {}".format((self.time_step * 4) / 1000))
        calcd_time_step = (self.time_step * 4) / 1000
        #calcd_time_step = self.time_step

        self.pitchPID.sample_time = calcd_time_step
        self.rollPID.sample_time = calcd_time_step
        self.yawPID.sample_time = calcd_time_step
        self.throttlePID.sample_time = calcd_time_step

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

            #write mavic2pro state
            mavic_state = [x_pos, y_pos, altitude, roll, pitch, yaw,
                           roll_acceleration, pitch_acceleration, yaw_acceleration,
                           self.getTime()]
            writeMavicState(mavic_state, files_dict['state-file'])

            print("at timestep: {}".format(self.getTime()))
            print("rPID: {:.4f} pPID: {:.4f} yPID: {:.4f} tPID: {:.4f}".format(self.rollPID(x_pos), 
                                                                  self.pitchPID(y_pos), 
                                                                  self.yawPID(yaw),
                                                                  self.throttlePID(altitude)))
            
            ###experimental waypoints
            waypoints = [[-8.0, 0.0]]
            ###check if we have reached target-altitude
            if altitude > self.target_altitude - 0.1:
                # as soon as it reach the target altitude, compute the disturbances to go to the given waypoints.
                if self.getTime() - t1 > 0.1:
                    #get yaw disturbance first before getting pitch
                    yaw_disturbance = self.move_to_target(waypoints)
            
            
            #collect inputs
            #roll_input = float(params["k_roll_p"]) * clamp(roll, -1, 1) + roll_acceleration + self.rollPID(x_pos)
            roll_input = float(params["k_roll_p"]) * roll + roll_acceleration + self.rollPID(x_pos)
            #roll_input = self.rollPID(x_pos)
            #pitch_input = float(params["k_pitch_p"]) * clamp(pitch, -1, 1) + pitch_acceleration + self.pitchPID(-y_pos)
            pitch_input = float(params["k_pitch_p"]) * pitch + pitch_acceleration + self.pitchPID(y_pos)
            #pitch_input = self.pitchPID(-y_pos)
            print("climbing to target altitude: {:.4f}".format(float(self.target_altitude)))
            diff_altitude = self.target_altitude - altitude + float(params["k_vertical_offset"])

            clamped_difference_altitude = clamp(self.target_altitude - altitude + float(params["k_vertical_offset"]), -1.0, 1.0)
            #clamped_difference_altitude = float(self.target_altitude) - altitude + float(params["k_vertical_offset"])
            #vertical_input = self.throttlePID(altitude) * pow(clamped_difference_altitude, 3)
            vertical_input = self.throttlePID(clamped_difference_altitude)


            #yaw_input = self.yawPID(yaw)
            k_yaw_p = 2.8
            self.yawPID.setpoint = yaw_disturbance
            #yaw_input = k_yaw_p * yaw + yaw_acceleration + self.yawPID(yaw)
            yaw_input = self.yawPID(yaw)
            
            
            
            
            
            print("inputs-->roll: {:.4f}, pitch: {:.4f}, yaw: {:.4f}, vertical: {:.4f}".format(roll_input, pitch_input, yaw_input, vertical_input))

            #roll_input = self.K_ROLL_P * clamp(roll, -1, 1) + roll_acceleration + roll_disturbance
            #pitch_input = self.K_PITCH_P * clamp(pitch, -1, 1) + pitch_acceleration + pitch_disturbance
            #yaw_input = yaw_disturbance
            #clamped_difference_altitude = clamp(self.target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1)
            #vertical_input = self.K_VERTICAL_P * pow(clamped_difference_altitude, 3.0)
            print("vert input: {}".format(vertical_input))

            #write PID and rotational and vertical inputs
            pid_inputs = [self.rollPID(x_pos), self.pitchPID(y_pos), 
                          self.yawPID(yaw), self.throttlePID(altitude),
                          roll_input, pitch_input, vertical_input, yaw_input, 
                          diff_altitude, clamped_difference_altitude]
            writePIDandInputs(pid_inputs, files_dict['pid-file'])


            front_left_motor_input = float(params["k_vertical_thrust"]) + vertical_input - yaw_input + pitch_input - roll_input
            front_right_motor_input = float(params["k_vertical_thrust"]) + vertical_input + yaw_input + pitch_input + roll_input
            rear_left_motor_input = float(params["k_vertical_thrust"]) + vertical_input + yaw_input - pitch_input - roll_input
            rear_right_motor_input = float(params["k_vertical_thrust"]) + vertical_input - yaw_input - pitch_input + roll_input

            #front_left_motor_input = float(params["k_vertical_thrust"]) + vertical_input - roll_input - pitch_input + yaw_input
            #front_right_motor_input = float(params["k_vertical_thrust"]) + vertical_input + roll_input - pitch_input - yaw_input
            #rear_left_motor_input = float(params["k_vertical_thrust"]) + vertical_input - roll_input + pitch_input - yaw_input
            #rear_right_motor_input = float(params["k_vertical_thrust"]) + vertical_input + roll_input + pitch_input + yaw_input

            print("motors:\n{:.4f}, {:.4f}, {:.4f}, {:.4f}".format(front_left_motor_input, front_right_motor_input, rear_left_motor_input, rear_right_motor_input))

            self.front_left_motor.setVelocity(front_left_motor_input)
            self.front_right_motor.setVelocity(-front_right_motor_input)
            self.rear_left_motor.setVelocity(-rear_left_motor_input)
            self.rear_right_motor.setVelocity(rear_right_motor_input)




#main function
def main():
    #clear output files if they exit
    filedir = r"C:\Users\ericx\OneDrive\Desktop\mine\grad_school\MARS_Lab_Swarm\Gazebo_dragonFly\webots_tutorial\hovering_test_project\python_utilities"
    ###experimental-designator filename
    state_filename = filedir + r"\mavic_state2.csv"
    pid_filename = filedir + r"\PID_and_inputs2.csv"
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
            
    TIME_STEP = int(params["QUADCOPTER_TIME_STEP"])
    TAKEOFF_THRESHOLD_VELOCITY = int(params["TAKEOFF_THRESHOLD_VELOCITY"])
    robot = Mavic(time_step=TIME_STEP,
                  params=params,
                  takeoff_threshold_velocity=TAKEOFF_THRESHOLD_VELOCITY)
    robot.run(files_dict, params=params)


if __name__ == '__main__':
    main()





