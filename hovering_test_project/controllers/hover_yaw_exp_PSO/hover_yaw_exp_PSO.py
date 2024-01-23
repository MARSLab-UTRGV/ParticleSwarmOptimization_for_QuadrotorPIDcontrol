"""hover_yaw_exp_PSO controller.
1. Start PSO for Yaw PID-params
2. Params Domain: 
    {'P': (0,2)
    {'I': (0.00001, 0.01)
    {'D': (0, 1)
3. PSO:
    pbest_obj-fcn = (ln(pos_ovrshot/pos_expect +1) + ln(time_ovrshot/time_expect +1))
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
    from csv import DictWriter
    from simple_pid import PID
except ImportError:
    sys.exit("Warning: 'numpy' module not found.")
#################################################

###Object-Class for Mavic Drone
class Mavic(Robot):

    def __init__(self, time_step, params, takeoff_threshold_velocity=1):
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
        xpos_setpoint = 1e-6
        ypos_setpoint = 1e-6
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
                          setpoint=float(self.yaw_setpoint))

        #enable current pose for robot
        self.current_pose = self.gps.getValues()  # X, Y, Z, yaw, pitch, roll
        #including previous-pose to calculate velocity
        self.previous_pose = 6 * [0]
        #target values
        #set target altitude from given z-waypoint
        try:
            #Drone moves to takeoff state
            self.target_position = [xpos_setpoint, ypos_setpoint, float(params["target_altitude"])]
            print("Using first waypt: {}".format(self.target_position))
            if self.target_position[2] > 0.5:
                self.target_altitude = float(self.target_position[2])
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
        self.throttlePID.setpoint = self.target_altitude###setting trottlePID setpoint to params
        print("Mavic Drone initialized...")


    #mavic helper-fcn
    def set_position(self, pos):
        """
        Set the new absolute position of the robot
        Parameters:
            pos (list): [X,Y,Z,yaw,pitch,roll] current absolute position and angles
        """
        self.current_pose = pos

        
    #mavic helper-fcn
    def set_previous_position(self, pos):
        self.previous_pose = pos

        
    #mavic helper-fcn
    def getVelocities(self, timestep):
        x_vel = (self.current_pose[0] - self.previous_pose[0]) / timestep
        y_vel = (self.current_pose[1] - self.previous_pose[1]) / timestep
        z_vel = (self.current_pose[2] - self.previous_pose[2]) / timestep
        #print("vels: {:.4f}, {:.4f}, {:.4f}".format(self.current_pose[0], self.previous_pose[0], timestep))
        #print("altitude vel: {:.4f}".format(z_vel))
        return x_vel, y_vel, z_vel


    #change yaw to move to first waypoint
    def chgYaw_to_target(self, params, tgt_waypt):
        self.target_position = tgt_waypt
        print("target waypt: {}".format(self.target_position))

        #calculate yaw_setpoint for target waypoint
        #This will be in ]-pi;pi]
        yaw_chg = np.arctan2(self.target_position[1] - self.current_pose[1],
                             self.target_position[0] - self.current_pose[0])
        print("current yaw: {:.4f}".format(self.current_pose[5]))
        print("calcd yaw chg: {:.4f}".format(yaw_chg))

        # This is now in ]-2pi;2pi[
        angle_left = yaw_chg - self.current_pose[5]
        print("angle-left: {:.4f}".format(angle_left))

        # Normalize turn angle to ]-pi;pi]
        angle_left = (angle_left + 2 * np.pi) % (2 * np.pi)
        print("norm angle-left: {:.4f}".format(angle_left))

        if (angle_left > np.pi):
            angle_left -= 2 * np.pi
            print("new angle left: {}".format(angle_left))

        if self.yaw_setpoint != angle_left and self.drone_state == "hovr":
            self.yaw_setpoint = angle_left
            self.yawPID.setpoint = self.yaw_setpoint
            self.drone_state = self.drone_states['D']
            if self.drone_state == "yawd":
                print("new yaw changed and set...")


    #Mavic-run function
    def run(self, params, waypoints):
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

            #if bot reaches altitude and pos-vels are at 0-->change yaw to waypt
            #check hover-cond
            hover_cond = [altitude >= self.target_altitude - 0.2,
                          checkVels(x_vel, y_vel, alt_vel),
                          self.drone_state == 'toff']
            if np.all(hover_cond) and self.drone_state != 'hovr':
                self.drone_state = self.drone_states['C']
                print("in Hover state and changing yaw...")
                self.waypoint_idx += 1
                self.next_waypt = waypoints[self.waypoint_idx]
                print("...next waypt: {}".format(self.next_waypt))
                self.chgYaw_to_target(params, self.next_waypt)

            #state_msg
            state_msg = struct.pack("4s", self.drone_state.encode('utf-8'))
            self.mvc_emtr.send(state_msg)

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





#clamp-fcn: return min-val b/w input max value
#and (the max b/w curr val and min val)
def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)


#check vels and if all are zero-->drone in hover-state
#init-->static velocity variable=vels at which drone hovers in place
def checkVels(x_vel, y_vel, alt_vel):
    vel_arr = np.array([x_vel < 1e-2 and x_vel > -1e-2,
                       y_vel < 1e-2 and y_vel > -1e-2,
                       alt_vel < 1e-2 and alt_vel > -1e-2])
    return vel_arr.all()


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


#main method
def main():
    #params_file = "params_edit.csv"
    params_file = "optim_edit.csv"

    #init and get rand params for PD and PID-experiments
    params = getParams(params_file)
    #for key, val in params.items():
    #    print("{}:{}".format(key, val))

    TIME_STEP = int(params["QUADCOPTER_TIME_STEP"])
    TAKEOFF_THRESHOLD_VELOCITY = int(params["TAKEOFF_THRESHOLD_VELOCITY"])

    #collect waypoints
    waypoints = collectWaypoints()
    #print(waypoints)

    # create the Robot instance.
    robot = Mavic(time_step=TIME_STEP,
                  params=params,
                  takeoff_threshold_velocity=TAKEOFF_THRESHOLD_VELOCITY)
    robot.run(params, waypoints)

    
if __name__ == '__main__':
    main()


