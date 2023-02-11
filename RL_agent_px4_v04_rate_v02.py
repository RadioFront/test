from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, sys
from pymavlink import mavutil
import threading
import math
import os

#import base64
#import imageio
#import IPython
#import matplotlib.pyplot as plt
import numpy as np
#import PIL.Image
#import pyvirtualdisplay
import reverb

import tensorflow as tf

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import py_driver
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

#####################
# parameters
target_elevation = 50
period_telemetry = 1/10
frequency_message = 10
period_set_position = 1/5
time_arm_mode = 1
time_settling = 10
stamp_state = time.time()
stamp_telemetry = time.time()
stamp_set_position = time.time()
delay_unit = 0.01
time_steady_state = 1
no_sample=1200
sample_period = 0.1
mode = "none"
state = "arm"
stampState=time.time()
stampMode=time.time()
vec_telemetry=[]
vec_pid_update=[]
#####################

#####################
# drone controller parameters
## Rate Controller
MC_ROLLRATE_K = 1.0 #0.3 - 3
MC_ROLLRATE_D = 0.003 #0.0004 - 0.01
MC_ROLLRATE_I = 0.2 #0.1 - 0.5
MC_PITCHRATE_K = 1.0 #0.3 - 3
MC_PITCHRATE_D = 0.003 #0.0004 - 0.01
MC_PITCHRATE_I = 0.2 #0.1 - 0.5
MC_YAWRATE_K = 1.0 #0.3 - 3
MC_YAWRATE_I = 0.1 #0.04 - 0.4
## Attitude Controller
MC_ROLL_P = 6.5 #1-14
MC_PITCH_P = 6.5 #1-14
MC_YAW_P = 2.8 #1-5
## Velocity Controller
MPC_XY_VEL_P_ACC = 1.8 #1.2-5
MPC_XY_VEL_I_ACC = 0.2 #0.2-10
MPC_XY_VEL_D_ACC = 0.2 #0.1-2
MPC_Z_VEL_P_ACC = 4.0 #2-15
MPC_Z_VEL_I_ACC = 2.0 #0.2-3
MPC_Z_VEL_D_ACC = 0.0 #0-2
## Position Controller
MPC_XY_P = 0.95 #0-2
MPC_Z_P = 1.0 #0-2
#####################

def distance_target(A):
    return math.sqrt( A[0]**2 + A[1]**2 + (A[2]+target_elevation)**2 )

def distance_geo(A,B):
    return math.sqrt( (A[0]-B[0])**2 + (A[1]-B[1])**2 + (A[2]-B[2])**2 )

def request_message_interval(message_id: int, frequency_hz: float):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        message_id, # The MAVLink message ID
        1e6 / frequency_hz, # The interval between two messages in microseconds. Set to -1 to disable and 0 to request default rate.
        0, 0, 0, 0, # Unused parameters
        0, # Target address of message stream (if message has target address fields). 0: Flight-stack default (recommended), 1: address of requestor, 2: broadcast.
    )

def set_target_depth_local(depth):
    master.mav.set_position_target_local_ned_send( #84
        int(1e3 * (time.time() - boot_time)), # ms since boot
        master.target_system, master.target_component,
        coordinate_frame=mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask=( # ignore everything except z position
            # mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
            # mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
            # mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_FORCE_SET |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        ), x=0, y=0, z=depth,
        vx=0, vy=0, vz=0, 
        afx=0, afy=0, afz=0, yaw=0, yaw_rate=0
    )

def request_message():
    request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS, frequency_message) #1
    request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_SET_MODE, frequency_message) #1
    request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT, frequency_message) #24
    request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_SCALED_IMU, frequency_message) #26
    request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, frequency_message) #30
    request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_QUATERNION, frequency_message) #31
    request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, frequency_message) #32
    request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, frequency_message) #33
    request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_SERVO_OUTPUT_RAW, frequency_message) #36
    request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD, frequency_message) #74
    request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_TARGET, frequency_message) #83
    request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_POSITION_TARGET_LOCAL_NED, frequency_message) #85
    request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_POSITION_TARGET_GLOBAL_INT, frequency_message) #87
    request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_ALTITUDE, frequency_message) #141
    request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_BATTERY_STATUS, frequency_message) #147
    request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_ESTIMATOR_STATUS, frequency_message) #230
    request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_VIBRATION, frequency_message) #241
    request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_UTM_GLOBAL_POSITION, frequency_message) #340
    request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_OPEN_DRONE_ID_LOCATION, frequency_message) #12901 

def setup_pid_controller():
    #####################
    # drone controller parameters
    ## Rate Controller
    MC_ROLLRATE_K = 1.0 #0.3 - 3
    MC_ROLLRATE_D = 0.003 #0.0004 - 0.01
    MC_ROLLRATE_I = 0.2 #0.1 - 0.5
    MC_PITCHRATE_K = 1.0 #0.3 - 3
    MC_PITCHRATE_D = 0.003 #0.0004 - 0.01
    MC_PITCHRATE_I = 0.2 #0.1 - 0.5
    MC_YAWRATE_K = 1.0 #0.3 - 3
    MC_YAWRATE_I = 0.1 #0.04 - 0.4
    ## Attitude Controller
    MC_ROLL_P = 6.5 #1-14
    MC_PITCH_P = 6.5 #1-14
    MC_YAW_P = 2.8 #1-5
    ## Velocity Controller
    MPC_XY_VEL_P_ACC = 1.8 #1.2-5
    MPC_XY_VEL_I_ACC = 0.2 #0.2-10
    MPC_XY_VEL_D_ACC = 0.2 #0.1-2
    MPC_Z_VEL_P_ACC = 4.0 #2-15
    MPC_Z_VEL_I_ACC = 2.0 #0.2-3
    MPC_Z_VEL_D_ACC = 0.0 #0-2
    ## Position Controller
    MPC_XY_P = 0.95 #0-2
    MPC_Z_P = 1.0 #0-2
    #####################
    ## Rate Controller
    #PITCH
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_PITCHRATE_K', MC_PITCHRATE_K, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_PITCHRATE_D', MC_PITCHRATE_D, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_PITCHRATE_I', MC_PITCHRATE_I, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    #ROLL
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_ROLLRATE_K', MC_ROLLRATE_K, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_ROLLRATE_D', MC_ROLLRATE_D, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_ROLLRATE_I', MC_ROLLRATE_I, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    #YAW
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_YAWRATE_K', MC_YAWRATE_K, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_YAWRATE_I', MC_YAWRATE_I, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    ## Attitude Controller
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_ROLL_P', MC_ROLL_P, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_PITCH_P', MC_PITCH_P, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_YAW_P', MC_YAW_P, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    ## Velocity Controller
    #XY
    master.mav.param_set_send(master.target_system, master.target_component, b'MPC_XY_VEL_P_ACC', MPC_XY_VEL_P_ACC, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MPC_XY_VEL_I_ACC', MPC_XY_VEL_I_ACC, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MPC_XY_VEL_D_ACC', MPC_XY_VEL_D_ACC, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    #Z
    master.mav.param_set_send(master.target_system, master.target_component, b'MPC_Z_VEL_P_ACC', MPC_Z_VEL_P_ACC, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MPC_Z_VEL_I_ACC', MPC_Z_VEL_I_ACC, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MPC_Z_VEL_D_ACC', MPC_Z_VEL_D_ACC, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    ## Position Controller
    master.mav.param_set_send(master.target_system, master.target_component, b'MPC_XY_P', MPC_XY_P, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MPC_Z_P', MPC_Z_P, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

def update_pid_controller_rate(action):
    ## Rate Controller
    MC_PITCHRATE_K=action[0]
    MC_PITCHRATE_D=action[1]
    MC_PITCHRATE_I=action[2]
    MC_ROLLRATE_K=action[3]
    MC_ROLLRATE_D=action[4]
    MC_ROLLRATE_I=action[5]
    MC_YAWRATE_K=action[6]
    MC_YAWRATE_I=action[7]
    #PITCH
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_PITCHRATE_K', MC_PITCHRATE_K, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_PITCHRATE_D', MC_PITCHRATE_D, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_PITCHRATE_I', MC_PITCHRATE_I, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    #ROLL
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_ROLLRATE_K', MC_ROLLRATE_K, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_ROLLRATE_D', MC_ROLLRATE_D, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_ROLLRATE_I', MC_ROLLRATE_I, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    #YAW
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_YAWRATE_K', MC_YAWRATE_K, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_YAWRATE_I', MC_YAWRATE_I, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

def update_pid_controller_velocity_pos(action):
    ## Velocity Controller
    MPC_XY_VEL_P_ACC = action[0] #1.8 #1.2-5
    MPC_XY_VEL_I_ACC = action[1] #0.2 #0.2-10
    MPC_XY_VEL_D_ACC = action[2] #0.2 #0.1-2
    MPC_Z_VEL_P_ACC = action[3] #4.0 #2-15
    MPC_Z_VEL_I_ACC = action[4] #2.0 #0.2-3
    MPC_Z_VEL_D_ACC = action[5] #0.0 #0-2
    ## Position Controller
    MPC_XY_P = action[6] #0.95 #0-2
    MPC_Z_P = action[7] #1.0 #0-2

    ## Velocity Controller
    #XY
    master.mav.param_set_send(master.target_system, master.target_component, b'MPC_XY_VEL_P_ACC', MPC_XY_VEL_P_ACC, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MPC_XY_VEL_I_ACC', MPC_XY_VEL_I_ACC, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MPC_XY_VEL_D_ACC', MPC_XY_VEL_D_ACC, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    #Z
    master.mav.param_set_send(master.target_system, master.target_component, b'MPC_Z_VEL_P_ACC', MPC_Z_VEL_P_ACC, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MPC_Z_VEL_I_ACC', MPC_Z_VEL_I_ACC, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MPC_Z_VEL_D_ACC', MPC_Z_VEL_D_ACC, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    ## Position Controller
    master.mav.param_set_send(master.target_system, master.target_component, b'MPC_XY_P', MPC_XY_P, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MPC_Z_P', MPC_Z_P, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

def update_pid_controller_all(action):
    ## Rate Controller
    MC_PITCHRATE_K=action[0]
    MC_PITCHRATE_D=action[1]
    MC_PITCHRATE_I=action[2]
    MC_ROLLRATE_K=action[3]
    MC_ROLLRATE_D=action[4]
    MC_ROLLRATE_I=action[5]
    MC_YAWRATE_K=action[6]
    MC_YAWRATE_I=action[7]
    ## Attitude Controller
    MC_ROLL_P = action[8] #6.5 #1-14
    MC_PITCH_P = action[9] #6.5 #1-14
    MC_YAW_P = action[10] #2.8 #1-5    ## Velocity Controller
    MPC_XY_VEL_P_ACC = action[11] #1.8 #1.2-5
    MPC_XY_VEL_I_ACC = action[12] #0.2 #0.2-10
    MPC_XY_VEL_D_ACC = action[13] #0.2 #0.1-2
    MPC_Z_VEL_P_ACC = action[14] #4.0 #2-15
    MPC_Z_VEL_I_ACC = action[15] #2.0 #0.2-3
    MPC_Z_VEL_D_ACC = action[16] #0.0 #0-2
    ## Position Controller
    MPC_XY_P = action[17] #0.95 #0-2
    MPC_Z_P = action[18] #1.0 #0-2

    #PITCH
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_PITCHRATE_K', MC_PITCHRATE_K, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_PITCHRATE_D', MC_PITCHRATE_D, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_PITCHRATE_I', MC_PITCHRATE_I, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    #ROLL
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_ROLLRATE_K', MC_ROLLRATE_K, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_ROLLRATE_D', MC_ROLLRATE_D, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_ROLLRATE_I', MC_ROLLRATE_I, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    #YAW
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_YAWRATE_K', MC_YAWRATE_K, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_YAWRATE_I', MC_YAWRATE_I, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    ## Attitude Controller
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_ROLL_P', MC_ROLL_P, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_PITCH_P', MC_PITCH_P, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MC_YAW_P', MC_YAW_P, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    ## Velocity Controller
    #XY
    master.mav.param_set_send(master.target_system, master.target_component, b'MPC_XY_VEL_P_ACC', MPC_XY_VEL_P_ACC, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MPC_XY_VEL_I_ACC', MPC_XY_VEL_I_ACC, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MPC_XY_VEL_D_ACC', MPC_XY_VEL_D_ACC, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    #Z
    master.mav.param_set_send(master.target_system, master.target_component, b'MPC_Z_VEL_P_ACC', MPC_Z_VEL_P_ACC, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MPC_Z_VEL_I_ACC', MPC_Z_VEL_I_ACC, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MPC_Z_VEL_D_ACC', MPC_Z_VEL_D_ACC, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    ## Position Controller
    master.mav.param_set_send(master.target_system, master.target_component, b'MPC_XY_P', MPC_XY_P, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    master.mav.param_set_send(master.target_system, master.target_component, b'MPC_Z_P', MPC_Z_P, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)


def get_telemetry_data():
    ack_msg = master.recv_match(type='COMMAND_ACK')
    observation=[]
    t_param='SCALED_IMU' #26
    try: 
        SCALED_IMU_xacc = master.messages[t_param].xacc
        SCALED_IMU_yacc = master.messages[t_param].yacc
        SCALED_IMU_zacc = master.messages[t_param].zacc
        SCALED_IMU_xgyro = master.messages[t_param].xgyro
        SCALED_IMU_ygyro = master.messages[t_param].ygyro
        SCALED_IMU_zgyro = master.messages[t_param].zgyro
        SCALED_IMU_data=[SCALED_IMU_xacc/16384,SCALED_IMU_yacc/16384,SCALED_IMU_zacc/16384,SCALED_IMU_xgyro/16384,SCALED_IMU_ygyro/16384,SCALED_IMU_zgyro/16384]
        observation.extend(SCALED_IMU_data) #count=6
    except:
        print(t_param, ":", 'No message received')

    t_param='ATTITUDE' #30
    try: 
        ATTITUDE_roll = master.messages[t_param].roll
        ATTITUDE_pitch = master.messages[t_param].pitch
        ATTITUDE_yaw = master.messages[t_param].yaw
        ATTITUDE_rollspeed = master.messages[t_param].rollspeed
        ATTITUDE_pitchspeed = master.messages[t_param].pitchspeed
        ATTITUDE_yawspeed = master.messages[t_param].yawspeed
        ATTITUDE_data=[ATTITUDE_roll/2,ATTITUDE_pitch/2,ATTITUDE_yaw/2,ATTITUDE_rollspeed/5,ATTITUDE_pitchspeed/5,ATTITUDE_yawspeed/5]
        observation.extend(ATTITUDE_data) #count=6+6=12
    except:
        print(t_param, ":", 'No message received') #6

    t_param='ATTITUDE_QUATERNION' #31
    try: 
        ATTITUDE_QUATERNION_q1 = master.messages[t_param].q1
        ATTITUDE_QUATERNION_q2 = master.messages[t_param].q2
        ATTITUDE_QUATERNION_q3 = master.messages[t_param].q3
        ATTITUDE_QUATERNION_q4 = master.messages[t_param].q4
        ATTITUDE_QUATERNION_rollspeed = master.messages[t_param].rollspeed
        ATTITUDE_QUATERNION_pitchspeed = master.messages[t_param].pitchspeed
        ATTITUDE_QUATERNION_yawspeed = master.messages[t_param].yawspeed
        ATTITUDE_QUATERNION_data=[ATTITUDE_QUATERNION_q1,ATTITUDE_QUATERNION_q2,ATTITUDE_QUATERNION_q3,ATTITUDE_QUATERNION_q4,ATTITUDE_QUATERNION_rollspeed/5,ATTITUDE_QUATERNION_pitchspeed/5,ATTITUDE_QUATERNION_yawspeed/5]
        observation.extend(ATTITUDE_QUATERNION_data) #count=12+7=19
    except:
        print(t_param, ":", 'No message received')

    t_param='LOCAL_POSITION_NED' #32
    try: 
        LOCAL_POSITION_NED_x = master.messages[t_param].x
        LOCAL_POSITION_NED_y = master.messages[t_param].y
        LOCAL_POSITION_NED_z = master.messages[t_param].z + 10
        LOCAL_POSITION_NED_vx = master.messages[t_param].vx
        LOCAL_POSITION_NED_vy = master.messages[t_param].vy
        LOCAL_POSITION_NED_vz = master.messages[t_param].vz
        LOCAL_POSITION_NED_data=[LOCAL_POSITION_NED_x/10,LOCAL_POSITION_NED_y/10,(LOCAL_POSITION_NED_z+target_elevation)/10,LOCAL_POSITION_NED_vx/10,LOCAL_POSITION_NED_vy/10,LOCAL_POSITION_NED_vz/10]
        observation.extend(LOCAL_POSITION_NED_data) #count=19+6=25
    except:
        print(t_param, ":", 'No message received')

    t_param='SERVO_OUTPUT_RAW' #36
    try: 
        SERVO_OUTPUT_RAW_servo1_raw = master.messages[t_param].servo1_raw - 1500
        SERVO_OUTPUT_RAW_servo2_raw = master.messages[t_param].servo2_raw - 1500
        SERVO_OUTPUT_RAW_servo3_raw = master.messages[t_param].servo3_raw - 1500
        SERVO_OUTPUT_RAW_servo4_raw = master.messages[t_param].servo4_raw - 1500
        SERVO_OUTPUT_RAW_data=[SERVO_OUTPUT_RAW_servo1_raw/500,SERVO_OUTPUT_RAW_servo2_raw/500,SERVO_OUTPUT_RAW_servo3_raw/500,SERVO_OUTPUT_RAW_servo4_raw/500]
        observation.extend(SERVO_OUTPUT_RAW_data) #count=25+4=29
    except:
        print(t_param, ":", 'No message received')

    t_param='VFR_HUD' #74
    try: 
        #VFR_HUD_airspeed = master.messages[t_param].airspeed
        VFR_HUD_groundspeed = master.messages[t_param].groundspeed
        VFR_HUD_heading = master.messages[t_param].heading
        VFR_HUD_throttle = master.messages[t_param].throttle
        #VFR_HUD_alt = master.messages[t_param].alt
        VFR_HUD_climb = master.messages[t_param].climb
        VFR_HUD_data=[VFR_HUD_groundspeed/10,VFR_HUD_heading/360,VFR_HUD_throttle/100,VFR_HUD_climb/10]
        observation.extend(VFR_HUD_data) #count=29+4=33
    except:
        print(t_param, ":", 'No message received')
        
    t_param='ATTITUDE_TARGET' #83
    try: 
        ATTITUDE_TARGET_q = master.messages[t_param].q
        ATTITUDE_TARGET_body_roll_rate = master.messages[t_param].body_roll_rate
        ATTITUDE_TARGET_body_pitch_rate = master.messages[t_param].body_pitch_rate
        ATTITUDE_TARGET_body_yaw_rate = master.messages[t_param].body_yaw_rate
        ATTITUDE_TARGET_thrust = master.messages[t_param].thrust
        ATTITUDE_TARGET_data=[ATTITUDE_TARGET_q[0],ATTITUDE_TARGET_q[1],ATTITUDE_TARGET_q[2],ATTITUDE_TARGET_q[3],ATTITUDE_TARGET_body_roll_rate/4,ATTITUDE_TARGET_body_pitch_rate/4,ATTITUDE_TARGET_body_yaw_rate/4,ATTITUDE_TARGET_thrust]
        observation.extend(ATTITUDE_TARGET_data) #count=33+8=41
    except:
        print(t_param, ":", 'No message received')

    t_param='ESTIMATOR_STATUS' #230
    try: 
        ESTIMATOR_STATUS_vel_ratio = master.messages[t_param].vel_ratio
        ESTIMATOR_STATUS_pos_horiz_ratio = master.messages[t_param].pos_horiz_ratio
        ESTIMATOR_STATUS_pos_vert_ratio = master.messages[t_param].pos_vert_ratio
        ESTIMATOR_STATUS_mag_ratio = master.messages[t_param].mag_ratio
        ESTIMATOR_STATUS_data=[ESTIMATOR_STATUS_vel_ratio/5,ESTIMATOR_STATUS_pos_horiz_ratio,ESTIMATOR_STATUS_pos_vert_ratio,ESTIMATOR_STATUS_mag_ratio]
        observation.extend(ESTIMATOR_STATUS_data) #count=41+4=45
    except:
        print(t_param, ":", 'No message received')

    t_param='UTM_GLOBAL_POSITION' #340
    try: 
        UTM_GLOBAL_POSITION_vx = master.messages[t_param].vx
        UTM_GLOBAL_POSITION_vy = master.messages[t_param].vy
        UTM_GLOBAL_POSITION_vz = master.messages[t_param].vz
        UTM_GLOBAL_POSITION_data=[UTM_GLOBAL_POSITION_vx/1000, UTM_GLOBAL_POSITION_vy/1000, UTM_GLOBAL_POSITION_vz/1000]
        observation.extend(UTM_GLOBAL_POSITION_data) #count=45+3=48
    except:
        print(t_param, ":", 'No message received')

    t_param='OPEN_DRONE_ID_LOCATION' #12901
    try: 
        OPEN_DRONE_ID_LOCATION_speed_horizontal = master.messages[t_param].speed_horizontal
        OPEN_DRONE_ID_LOCATION_speed_vertical = master.messages[t_param].speed_vertical
        OPEN_DRONE_ID_LOCATION_data=[OPEN_DRONE_ID_LOCATION_speed_horizontal/1000, OPEN_DRONE_ID_LOCATION_speed_vertical/1000]
        observation.extend(OPEN_DRONE_ID_LOCATION_data) #count=48+2=50
    except:
        print(t_param, ":", 'No message received')

    return observation

def get_position_data():
    ack_msg = master.recv_match(type='COMMAND_ACK')
    t_param='LOCAL_POSITION_NED' #32
    try: 
        LOCAL_POSITION_NED_x = master.messages[t_param].x
        LOCAL_POSITION_NED_y = master.messages[t_param].y
        LOCAL_POSITION_NED_z = master.messages[t_param].z
        return [LOCAL_POSITION_NED_x, LOCAL_POSITION_NED_y, LOCAL_POSITION_NED_z]
    except:
        print(t_param, ":", 'No message received')
        return [0,0,0]

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def collect_episode(environment, policy, num_episodes):

    driver = py_driver.PyDriver(
        environment,
        py_tf_eager_policy.PyTFEagerPolicy(
        policy, use_tf_function=True),
        [rb_observer],
        max_episodes=num_episodes)
    initial_time_step = environment.reset()
    driver.run(initial_time_step)

class PX4_RL_PID(py_environment.PyEnvironment):

    def __init__(self, noSample):
        self.sample_size = noSample
        self.noObs = 50
        self.sample_stamp = 0
        self.track_distance = 0
        self.max_distance = 0
        self.action_history = []
        self.observation_history = []
        self.reward_accumulate = 0
        tActionMin=[0.3, 0.0004, 0.1, 0.3, 0.0004, 0.1, 0.3, 0.04] #, 1.0, 1.0, 1.0, 1.2, 0.2, 0.1, 2.0, 0.2, 0.0, 0.0, 0.0]
        tActionMax=[3.0, 0.01  , 0.5, 3.0, 0.01,   0.5, 3.0, 0.4 ] #, 14.0, 14.0, 5.0, 5.0, 10.0, 2.0, 15.0, 3.0, 2.0, 2.0, 2.0]
        tObsMin=[-1.0] * self.noObs
        tObsMax=[ 1.0] * self.noObs
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(8,), dtype=np.float32, minimum=tActionMin, maximum=tActionMax, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.noObs,), dtype=np.float32, minimum=tObsMin, maximum=tObsMax, name='observation')
        self._state = 0
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0
        self._episode_ended = False
        self.sample_stamp = 0
        self.track_distance = 0
        self.max_distance = 0
        self.action_history = []
        self.observation_history = []
        self.reward_accumulate = 0
        return ts.restart(np.array([0.0] * self.noObs, dtype=np.float32))

    def _step(self, action):
        sample_period=0.1
        self.action_history.append(action)

        update_pid_controller_rate(action)
        #update_pid_controller_all(action)
        #print("Action:", action)
        if self.sample_stamp == 0:
            self.sample_stamp = time.time()

        #wait for sample period
        while time.time() - self.sample_stamp < sample_period:
            time.sleep(0.01)
        self.sample_stamp = time.time()
    
        set_target_depth_local(-target_elevation)
        observation = get_telemetry_data()
        self.observation_history.append(observation)

        position = get_position_data()
        #print("Position:", self._state, position)
        distance = distance_target(position)
        self.track_distance  += distance
        if self.max_distance < distance:
            self.max_distance = distance

        if self._episode_ended:
            return self.reset()

        # Make sure episodes don't go on forever.
        #if action == 1:
        #    self._episode_ended = True
        #elif action == 0:
        #    new_card = np.random.randint(1, 11)
        #    self._state += new_card
        #else:
        #    raise ValueError('`action` should be 0 or 1.')
        #saction=[0.3, 0.0004, 0.1, 0.3, 0.0004, 0.1, 0.3, 0.04]
        self._state += 1
        if self._state > self.sample_size:
            self._episode_ended = True

        #current_reward = 1/(self.track_distance/self.sample_size)/self.max_distance
        current_reward = 1/(0.1+distance)/(0.1+self.max_distance)
        self.reward_accumulate += current_reward

        if self._episode_ended == True:
            
            reward = current_reward
            print("Distance: %.2f / Max distance: %.2f / Reward: %.3f, " % (self.track_distance/self.sample_size, self.max_distance, self.reward_accumulate))
            f=open("px4_rl_output"+str(self.sample_size)+".txt", 'a')
            f.write("%.3f\t%.3f\t%.3f\n" % (self.track_distance/self.sample_size, self.max_distance, self.reward_accumulate))
            f.close()
            f=open("px4_rl_action"+str(self.sample_size)+".txt", 'w')
            for items in self.action_history:
                for item in items:
                    f.write("%.3f\t" % item)
                f.write("\n")
            f.close()            
            f=open("px4_rl_observation"+str(self.sample_size)+".txt", 'w')
            for items in self.observation_history:
                for item in items:
                    f.write("%.3f\t" % item)
                f.write("\n")
            f.close()            
            return ts.termination(np.array(observation, dtype=np.float32), reward)
        else:
            return ts.transition(
                np.array(observation, dtype=np.float32), reward=current_reward, discount=1.0)

print("-----Setup PX4----------")
master = mavutil.mavlink_connection('udpin:localhost:14540')
boot_time = time.time()
master.wait_heartbeat()
print("Heartbeat: (system %u component %u)" % (master.target_system, master.target_component))
setup_pid_controller()

# Arm motors
print("State:",state)
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0,
    1, 0, 0, 0, 0, 0, 0)
master.motors_armed_wait()
state = "set_mode"
print("State:",state)
stampState = time.time()
stampMode = time.time()
request_message()
returns = []

##### Put down at steady state
while mode!="steady_state":
    # set OFFBOARD mode
    if state == "set_mode" and ( time.time() - stampState > time_arm_mode ):
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            209, 6, 0, 0, 0, 0, 0) #OFFBOARD mode
        state = "mode_wait"
        print("State:",state)
        stampState = time.time()

    #wait fof OFFBOARD mode acknowledged
    if state == "mode_wait" and ( time.time() - stampState > time_arm_mode):
        ack_msg = master.recv_match(type='COMMAND_ACK')
        if ack_msg:
            ack_msg = ack_msg.to_dict()
            if ack_msg['command'] == mavutil.mavlink.MAV_CMD_DO_SET_MODE:
                print(mavutil.mavlink.enums['MAV_RESULT'][ack_msg['result']].description)
                state = "ramp_up"
                stampState = time.time()
                print("State:",state)
                
    # wait till the drone reaches target position
    if state == "ramp_up" and ( time.time() - stampState > time_settling):
        state = "flight_mode"
        print("State:",state)
        stampState = time.time()

    # repeat position setup > 2 Hz, to avoid fail safe mode kick in
    if time.time() - stamp_set_position > period_set_position:
        stamp_set_position = time.time()
        set_target_depth_local(-target_elevation)

    # unit delay
    time.sleep(delay_unit)

    t_param='POSITION_TARGET_LOCAL_NED' #85
    try: 
        POSITION_TARGET_LOCAL_NED_x= master.messages[t_param].x
        #POSITION_TARGET_LOCAL_NED_y= master.messages[t_param].y
        #POSITION_TARGET_LOCAL_NED_z= master.messages[t_param].z
        #print(t_param, ":", POSITION_TARGET_LOCAL_NED_x,POSITION_TARGET_LOCAL_NED_y,POSITION_TARGET_LOCAL_NED_z)
        if mode == "none" and math.isnan(POSITION_TARGET_LOCAL_NED_x) == False:
            mode = "flight_mode"
            stampMode = time.time()
            print("Mode:",mode)
    except:
        print(t_param,":",'No message received')

    t_param='LOCAL_POSITION_NED' #32
    try: 
        LOCAL_POSITION_NED_x = master.messages[t_param].x
        LOCAL_POSITION_NED_y = master.messages[t_param].y
        LOCAL_POSITION_NED_z = master.messages[t_param].z
        #print(t_param, ":", LOCAL_POSITION_NED_x, LOCAL_POSITION_NED_y, LOCAL_POSITION_NED_z)
        if mode == "flight_mode" and distance_target([LOCAL_POSITION_NED_x, LOCAL_POSITION_NED_y, LOCAL_POSITION_NED_z])<0.3:
            mode = "steady_state"
            stampMode = time.time()
            print("Mode:",mode)
    except:
        print(t_param, ":", 'No message received')
##########################################################

print("-----Instantiate PID----------")

train_py_env = PX4_RL_PID(600)
eval_py_env = PX4_RL_PID(1200)
num_iterations = 1000 # @param {type:"integer"}
collect_episodes_per_iteration = 2 # @param {type:"integer"}
replay_buffer_capacity = 5000 # @param {type:"integer"}

fc_layer_params = (200,400,200,100,50)

learning_rate = 1e-3 # @param {type:"number"}
log_interval = 25 # @param {type:"integer"}
num_eval_episodes = 3# @param {type:"integer"}
eval_interval = 50 # @param {type:"integer"}

print("-----TF setup----------")

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.compat.v1.train.get_or_create_global_step()

#train_step_counter = tf.Variable(0)

#print(train_env.time_step_spec())
#print(train_env.action_spec())

print("-----TF agent----------")
tf_agent = reinforce_agent.ReinforceAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_network=actor_net,
    optimizer=optimizer,
    normalize_returns=True,
    train_step_counter=train_step_counter)

tf_agent.initialize()

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

print("-----Reverve----------")
table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(
    tf_agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)
table = reverb.Table(
    table_name,
    max_size=replay_buffer_capacity,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    tf_agent.collect_data_spec,
    table_name=table_name,
    sequence_length=None,
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddEpisodeObserver(
    replay_buffer.py_client,
    table_name,
    replay_buffer_capacity
)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
tf_agent.train = common.function(tf_agent.train)

# Reset the train step
tf_agent.train_step_counter.assign(0)

print("-----Train----------")

print("-----Evaluate----------")
# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
returns = [avg_return]

print(train_step_counter)
tempdir='.'
checkpoint_dir = os.path.join(tempdir, 'checkpoint')
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=tf_agent,
    policy=tf_agent.policy,
    replay_buffer=replay_buffer,
    global_step=train_step_counter
    )

train_checkpointer.initialize_or_restore()
#train_step_counter = tf.compat.v1.train.get_global_step()
print(train_step_counter)

############################################################
for _ in range(num_iterations):
    print("Iteration:", _, train_step_counter)
    # Put RWAV at steady state position before data collection starts
    #########################################################
    # Arm motors
    #print("State:",state)
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, 0, 0, 0, 0, 0, 0)
    master.motors_armed_wait()
    state = "set_mode"
    #print("State:",state)
    stampState = time.time()
    stampMode = time.time()
    #request_message()
    setup_pid_controller()

    mode="none"
    ##### Put down at steady state
    while mode!="steady_state":
        # set OFFBOARD mode
        if state == "set_mode" and ( time.time() - stampState > time_arm_mode ):
            master.mav.command_long_send(
                master.target_system,
                master.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,
                209, 6, 0, 0, 0, 0, 0) #OFFBOARD mode
            state = "mode_wait"
            print("State:",state)
            stampState = time.time()

        #wait fof OFFBOARD mode acknowledged
        if state == "mode_wait" and ( time.time() - stampState > time_arm_mode):
            ack_msg = master.recv_match(type='COMMAND_ACK')
            if ack_msg:
                ack_msg = ack_msg.to_dict()
                if ack_msg['command'] == mavutil.mavlink.MAV_CMD_DO_SET_MODE:
                    print(mavutil.mavlink.enums['MAV_RESULT'][ack_msg['result']].description)
                    state = "ramp_up"
                    stampState = time.time()
                    print("State:",state)
                    
        # wait till the drone reaches target position
        if state == "ramp_up" and ( time.time() - stampState > time_settling):
            state = "flight_mode"
            print("State:",state)
            stampState = time.time()

        # repeat position setup > 2 Hz, to avoid fail safe mode kick in
        if time.time() - stamp_set_position > period_set_position:
            stamp_set_position = time.time()
            set_target_depth_local(-target_elevation)

        # unit delay
        time.sleep(delay_unit)

        t_param='POSITION_TARGET_LOCAL_NED' #85
        try: 
            POSITION_TARGET_LOCAL_NED_x= master.messages[t_param].x
            if mode == "none" and math.isnan(POSITION_TARGET_LOCAL_NED_x) == False:
                mode = "flight_mode"
                stampMode = time.time()
                print("Mode:",mode)
        except:
            print(t_param,":",'No message received')

        t_param='LOCAL_POSITION_NED' #32
        try: 
            LOCAL_POSITION_NED_x = master.messages[t_param].x
            LOCAL_POSITION_NED_y = master.messages[t_param].y
            LOCAL_POSITION_NED_z = master.messages[t_param].z
            if mode == "flight_mode" and distance_target([LOCAL_POSITION_NED_x, LOCAL_POSITION_NED_y, LOCAL_POSITION_NED_z])<0.3:
                mode = "steady_state"
                stampMode = time.time()    
                print("Mode:",mode, " - %.2f, %.2f, %.2f" % (LOCAL_POSITION_NED_x, LOCAL_POSITION_NED_y, LOCAL_POSITION_NED_z))       
        except:
            print(t_param, ":", 'No message received')
    ##################################################################################

    # Collect a few episodes using collect_policy and save to the replay buffer.
    collect_episode(
        train_py_env, tf_agent.collect_policy, collect_episodes_per_iteration)

    # Use data from the buffer and update the agent's network.
    iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
    trajectories, _ = next(iterator)
    train_loss = tf_agent.train(experience=trajectories)  

    replay_buffer.clear()

    step = tf_agent.train_step_counter.numpy()
    
    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)
        
    train_checkpointer.save(train_step_counter)

    tempdir='.'
checkpoint_dir = os.path.join(tempdir, 'checkpoint')
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=tf_agent,
    policy=tf_agent.policy,
    replay_buffer=replay_buffer,
    global_step=train_step_counter
    )
train_checkpointer.save(train_step_counter)
print(train_step_counter)

print("-----Evaluate----------")
# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, tf_agent.policy, 31)
print(avg_return)