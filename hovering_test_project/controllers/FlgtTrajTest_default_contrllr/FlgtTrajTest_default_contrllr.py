"""Flight Trajectory Test for Default Controller.
Default Controller taken from: 
https://github.com/cyberbotics/webots-projects/blob/master/projects/forest_firefighters/controllers/autonomous_mavic/autonomous_mavic.py
Using similar approach to travel to waypoints and measure MSE
1. Starting Postion-->(0, 0, 0.116)
2. Achieve take-off state to target altitude
3. Hover at target-altitude
4. Once vels==0, change yaw rotation to first waypoint
5. Chg-Yaw towards next target waypoint
6. Change pitch to move towards target
7. Repeat for remaing target waypoints
8. Collect MSE from ref-traj to flight-traj
9. Later, compare with PID-controller performance
"""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
###IMPORTS#######################################
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


#clamp-fcn: return min-val b/w input max value
#and (the max b/w curr val and min val)
def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)


#clear file, if exists
def clearFileIfExists(filename):
    if os.path.isfile(filename):
        os.remove(filename)


def main():
    #clear output files if they exit
    filedir = r"C:\Users\ericx\OneDrive\Desktop\mine\grad_school\MARS_Lab_Swarm\Gazebo_dragonFly\webots_tutorial\hovering_test_project\python_utilities"
    ###experimental-designator filename
    state_filename = filedir + r"\mavic_stateDefault.csv"
    inputs_filename = filedir + r"\Default_and_inputs.csv"
    clearFileIfExists(state_filename)
    clearFileIfExists(inputs_filename)
    files_dict = {'state-file': state_filename, 'pid-file': inputs_filename}
    

    



if __name__ == '__main__':
    main()



