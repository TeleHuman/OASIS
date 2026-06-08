import argparse
import json
import time

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import mujoco as mj
import mujoco.viewer as mjv
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import draw_frame
from tqdm import tqdm
import redis
from rich import print
from general_motion_retargeting import XRobotStreamer

from tasks.params import DEFAULT_MIMIC_OBS, DEFAULT_HAND_POSE, ROBOT_XML_DICT, ROBOT_BASE_DICT, REDIS_ACTION_KEY
from data_utils.fps_monitor import FPSMonitor


def _root_ang_vel_w(cur_quat_wxyz, prev_quat_wxyz, dt):
    """World-frame angular velocity from wxyz quats using scipy Rotation."""
    # scipy uses xyzw; convert wxyz -> xyzw via np.roll(..., -1).
    cur = R.from_quat(np.roll(np.asarray(cur_quat_wxyz, dtype=np.float64), -1))
    prev = R.from_quat(np.roll(np.asarray(prev_quat_wxyz, dtype=np.float64), -1))
    return ((cur * prev.inv()).as_rotvec() / dt).astype(np.float64)


def extract_mimic_obs(qpos, last_qpos, dt=1/30):
    """Extract whole body mimic observations from robot joint positions (35 dims)"""
    root_pos = qpos[0:3]
    root_quat = qpos[3:7]
    joint_pos = qpos[7:].copy()  # Make a copy to avoid modifying original
    joint_vel = (qpos[7:] - last_qpos[7:]) / dt

    mimic_obs = {
        "joint_pos": joint_pos.tolist(), # 形状: (29,)
        "joint_vel": joint_vel.tolist(), # 形状: (29,)
        "root_pos": root_pos.tolist(),   # 形状: (3,)
        "root_quat": root_quat.tolist(),  # 形状: (4,)
    }
    return mimic_obs

class StateMachine:
    def __init__(self):
        self.state = "teleop"
        self.previous_state = "teleop"
        self.right_key_one_was_pressed = False
        self.left_key_one_was_pressed = False
        self.right_key_two_was_pressed = False
        self.left_key_two_was_pressed = False
        self.left_axis_click_was_pressed = False

        # Hand state - interpolation values (0.0 = open, 1.0 = closed)
        self.hand_left_position = 0.0  # 0.0 = fully open, 1.0 = fully closed
        self.hand_right_position = 0.0
        # Hand control parameters
        self.hand_movement_step = 0.05  # 5% movement per press/hold
        
        # start record and save
        self.start_record = False
        self.start_save = False
        self.reset_robot = False


    def update(self, controller_data):
        """Update state machine with controller data"""
        # Store previous state
        self.previous_state = self.state
        
        # Get current button states
        right_key_current = controller_data.get('RightController', {}).get('key_one', False)
        left_key_current = controller_data.get('LeftController', {}).get('key_one', False)

        right_key_two_current = controller_data.get('RightController', {}).get('key_two', False)
        left_key_two_current = controller_data.get('LeftController', {}).get('key_two', False)
        
        # Hand control - index_trig for close, grip for open
        right_index_trig_current = controller_data.get('RightController', {}).get('index_trig', False)
        left_index_trig_current = controller_data.get('LeftController', {}).get('index_trig', False)
        right_grip_current = controller_data.get('RightController', {}).get('grip', False)
        left_grip_current = controller_data.get('LeftController', {}).get('grip', False)

        # Emergency stop - left controller axis_click
        left_axis_click_current = controller_data.get('LeftController', {}).get('axis_click', False)

        # Detect button presses
        right_key_just_pressed = right_key_current and not self.right_key_one_was_pressed
        left_key_just_pressed = left_key_current and not self.left_key_one_was_pressed
        right_key_two_just_pressed = right_key_two_current and not self.right_key_two_was_pressed
        left_key_two_just_pressed = left_key_two_current and not self.left_key_two_was_pressed
        left_axis_click_just_pressed = left_axis_click_current and not self.left_axis_click_was_pressed


        # Right A (key_one): reset robot. (B / key_two is intercepted by the
        # PICO XRoboToolkit APK as a stream display-mode toggle, so it cannot
        # be used here.)
        if right_key_just_pressed:
            self.reset_robot = True
            self.hand_left_position = 0.0
            self.hand_right_position = 0.0
        else:
            self.reset_robot = False

        if left_key_just_pressed:
            self.start_record = True
        else:
            self.start_record = False

        if left_key_two_just_pressed:
            self.start_save = True
        else:
            self.start_save = False

        # Handle hand control - continuous interpolation
        # Right hand control
        if right_index_trig_current:  # Close right hand
            new_position = min(1.0, self.hand_right_position + self.hand_movement_step)
            if new_position != self.hand_right_position:
                self.hand_right_position = new_position
                print(f"Right hand closing: {self.hand_right_position:.1f}")
        elif right_grip_current:  # Open right hand
            new_position = max(0.0, self.hand_right_position - self.hand_movement_step)
            if new_position != self.hand_right_position:
                self.hand_right_position = new_position
                print(f"Right hand opening: {self.hand_right_position:.1f}")
        
        # Left hand control
        if left_index_trig_current:  # Close left hand
            new_position = min(1.0, self.hand_left_position + self.hand_movement_step)
            if new_position != self.hand_left_position:
                self.hand_left_position = new_position
                print(f"Left hand closing: {self.hand_left_position:.1f}")
        elif left_grip_current:  # Open left hand
            new_position = max(0.0, self.hand_left_position - self.hand_movement_step)
            if new_position != self.hand_left_position:
                self.hand_left_position = new_position
                print(f"Left hand opening: {self.hand_left_position:.1f}")
        
        
        # Update button state tracking
        self.right_key_one_was_pressed = right_key_current
        # self.left_key_one_was_pressed = left_key_current
        # self.left_key_two_was_pressed = left_key_two_current
        # self.right_key_two_was_pressed = right_key_two_current
        self.left_axis_click_was_pressed = left_axis_click_current
        
    
    def get_current_state(self):
        return self.state
    
    def get_hand_state(self):
        return self.hand_left_position, self.hand_right_position
    
    def get_hand_pose(self, robot_name):
        """Get interpolated hand poses based on current hand positions"""
        
        left_open = DEFAULT_HAND_POSE[robot_name]['left']['open']
        left_closed = DEFAULT_HAND_POSE[robot_name]['left']['close']
        right_open = DEFAULT_HAND_POSE[robot_name]['right']['open']
        right_closed = DEFAULT_HAND_POSE[robot_name]['right']['close']
        
        # Interpolate between open and closed poses
        left_pose = left_open + (left_closed - left_open) * self.hand_left_position
        right_pose = right_open + (right_closed - right_open) * self.hand_right_position
        
        return left_pose, right_pose
    

class XRobotTeleopToRobot:
    def __init__(self, args):
        self.args = args
        self.robot_name = args.robot
        self.xml_file = ROBOT_XML_DICT[args.robot]
        self.robot_base = ROBOT_BASE_DICT[args.robot]
        # Initialize state tracking
        self.last_qpos = None
        self.last_time = time.time()
        self.target_fps = args.target_fps
        self.measured_dt = 1/ self.target_fps # default fallback dt

        # Initialize components
        self.teleop_data_streamer = None
        self.redis_client = None
        self.retarget = None
        self.model = None
        self.data = None
        self.state_machine = StateMachine()
        self.rate = None
                
        # FPS monitoring
        self.fps_monitor = FPSMonitor(
            enable_detailed_stats=args.measure_fps,
            quick_print_interval=100,
            detailed_print_interval=1000,
            expected_fps=self.target_fps,
            name="Teleop Loop"
        )


    def setup_teleop_data_streamer(self):
        """Initialize and start the teleop data streamer"""
        self.teleop_data_streamer = XRobotStreamer()
        print("Teleop data streamer initialized")
        
    def setup_redis_connection(self):
        """Setup Redis connection"""
        redis_ip = self.args.redis_ip
        self.redis_client = redis.Redis(host=redis_ip, port=6379, db=0, password="zjk1217...")
        self.redis_pipeline = self.redis_client.pipeline()
        self.redis_client.ping()
        print("Redis connected successfully")

    def setup_retargeting_system(self):
        """Initialize the motion retargeting system"""
        self.retarget = GMR(
            src_human="xrobot",
            tgt_robot="unitree_g1",
            actual_human_height=self.args.actual_human_height,
        )
        print("Retargeting system initialized")
    
    def setup_mujoco_simulation(self):
        """Setup MuJoCo model and data"""
        self.model = mj.MjModel.from_xml_path(str(self.xml_file))
        self.data = mj.MjData(self.model)
        print("MuJoCo simulation initialized")
        
    def setup_rate_limiter(self):
        """Setup rate limiter for consistent FPS"""
        self.rate = RateLimiter(frequency=self.target_fps, warn=False)
        print(f"Rate limiter setup for {self.target_fps} FPS")
        
    def get_teleop_data(self):
        """Get current teleop data from streamer"""
        if self.teleop_data_streamer is not None:
            return self.teleop_data_streamer.get_current_frame()
        return None, None, None, None, None
        
    def process_retargeting(self, smplx_data):
        """Process motion retargeting and return observations"""
        if smplx_data is None or self.retarget is None:
            return None, None
            
        # Measure dt between retarget calls
        current_time = time.time()
        self.measured_dt = current_time - self.last_time
        self.last_time = current_time
        
        # Retarget till convergence
        qpos = self.retarget.retarget(smplx_data, offset_to_ground=True)
        
        # Create mimic obs from retargeting
        if self.last_qpos is not None:
            current_retarget_obs = extract_mimic_obs(qpos, self.last_qpos, dt=self.measured_dt)
        else:
            current_retarget_obs = DEFAULT_MIMIC_OBS[self.robot_name]
        
        self.last_qpos = qpos.copy()
        return qpos, current_retarget_obs
        
    def update_visualization(self, qpos, smplx_data, viewer):
        """Update MuJoCo visualization"""
        if qpos is None:
            return

        # Clean custom geometry
        if hasattr(viewer, 'user_scn') and viewer.user_scn is not None:
            viewer.user_scn.ngeom = 0
            
        # Draw the task targets for reference
        if smplx_data is not None and self.retarget is not None:
            for robot_link, ik_data in self.retarget.ik_match_table1.items():
                body_name = ik_data[0]
                if body_name not in smplx_data:
                    continue
                draw_frame(
                    self.retarget.scaled_human_data[body_name][0] - self.retarget.ground,
                    R.from_quat(smplx_data[body_name][1]).as_matrix(),
                    viewer,
                    0.1,
                    orientation_correction=R.from_quat(ik_data[-1]),
                )
                
        self.data.qpos[:] = qpos.copy()
        mj.mj_forward(self.model, self.data)
        
        # Camera follow the pelvis
        self._update_camera_position(viewer)
        
    def _update_camera_position(self, viewer):
        """Update camera to follow the robot"""
        FOLLOW_CAMERA = True
        if FOLLOW_CAMERA:
            robot_base_pos = self.data.xpos[self.model.body(self.robot_base).id]
            viewer.cam.lookat = robot_base_pos
            viewer.cam.distance = 3.0
            
            

            
    def determine_mimic_obs_to_send(self, current_retarget_obs):                              
      if self.state_machine.state == "idle" or current_retarget_obs is None:              
          return DEFAULT_MIMIC_OBS[self.robot_name]                                         
      return current_retarget_obs  
    
            
    def send_to_redis(self, mimic_obs):
        """Send mimic observations to Redis"""
        if self.state_machine.state == "idle":
            # In idle state, we choose to skip sending to keep the queue clean.
            return
        if self.redis_client is not None and mimic_obs is not None:
            combined_key = REDIS_ACTION_KEY
            # if self.state_machine.state == "teleop" and self.state_machine.previous_state == "idle":
            #     self.redis_pipeline.delete(combined_key)
            #     print("Delete old msg.Start to teleop") 

            hand_left_pose, hand_right_pose = self.state_machine.get_hand_pose(self.robot_name)

            mimic_obs_with_t = dict(mimic_obs)
            mimic_obs_with_t["t"] = time.time()
            combined_action = {
                "body": mimic_obs_with_t,
                "left_hand": hand_left_pose.tolist(),
                "right_hand": hand_right_pose.tolist(),
                "start_record": self.state_machine.start_record,
                "start_save": self.state_machine.start_save,
                "reset_robot": self.state_machine.reset_robot,
            }        
            # self.redis_pipeline.rpush(combined_key, json.dumps(combined_action))
            # self.redis_pipeline.ltrim(combined_key, -50, -1)  # 保持队列新鲜，防止堆积
            self.redis_pipeline.set(combined_key, json.dumps(combined_action))
            self.redis_pipeline.execute()        
            


    def initialize_all_systems(self):
        """Initialize all required systems"""
        print("Initializing teleop systems...")
        self.setup_teleop_data_streamer()
        self.setup_redis_connection()
        self.setup_retargeting_system()
        self.setup_mujoco_simulation()
        self.setup_rate_limiter()

        print("Teleop state machine initialized. Controls:")
        print("- Right controller key_one: Cycle through idle -> teleop -> idle...")
        print("- Right controller key_two: Reset")
        print("- Left controller key_one: Start Record")
        print("- Left controller key_two: Start Save")
        print(f"Starting in state: {self.state_machine.get_current_state()}")

        
        if self.fps_monitor.enable_detailed_stats:
            print(f"- FPS measurement: ENABLED (detailed stats every {self.fps_monitor.detailed_print_interval} steps)")
        else:
            print(f"- FPS measurement: Quick stats only (every {self.fps_monitor.quick_print_interval} steps)")

        print("Ready to receive teleop data.")

    def run(self):
        """Main execution loop"""
        self.initialize_all_systems()
        
        # Start the viewer
        with mjv.launch_passive(
            model=self.model, 
            data=self.data, 
            show_left_ui=False, 
            show_right_ui=False
        ) as viewer:
            viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 1
            
            while viewer.is_running():
                # Get current teleop data
                smplx_data, left_hand_data, right_hand_data, controller_data, headset_data = self.get_teleop_data()
                
                # Update state machine
                if controller_data is not None:
                    self.state_machine.update(controller_data)
                
                # Process retargeting if we have data
                qpos, current_retarget_obs = None, None
                if smplx_data is not None:
                    qpos, current_retarget_obs = self.process_retargeting(smplx_data)
                    self.update_visualization(qpos, smplx_data, viewer)
                
                
                # Determine and send mimic observations
                mimic_obs_to_send = self.determine_mimic_obs_to_send(current_retarget_obs)
                
                self.send_to_redis(mimic_obs_to_send)
                
                # Update visualization and record video
                viewer.sync()
                
                # FPS monitoring
                self.fps_monitor.tick()
                
                self.rate.sleep()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot",
        choices=["unitree_g1"],
        default="unitree_g1",
    )
    parser.add_argument(
        "--redis_ip",
        type=str,
        default="localhost",
        help="Redis IP",
    )
    parser.add_argument(
        "--actual_human_height",
        type=float,
        default=1.5,
        help="Actual human height for retargeting.",
    )   
    parser.add_argument(
        "--target_fps",
        type=int,
        default=100,
        help="Target FPS for the teleop system.",
    )
    parser.add_argument(
        "--measure_fps",
        type=int,
        default=0,
        help="Measure and print detailed FPS statistics (0=disabled, 1=enabled).",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    teleop_robot = XRobotTeleopToRobot(args)
    teleop_robot.run()