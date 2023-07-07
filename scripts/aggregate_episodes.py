#!/usr/bin/env python

from rosbag import Bag
import rospy
import tf2_ros

from tams_pr2_guzheng.msg import (
    PluckEpisodeV2,
    BiotacStamped,
    EpisodeState,
    ActionParameters,
    ExecutePathActionGoal,
    ExecutePathActionResult,
    TactilePluck,
    )
from music_perception.msg import NoteOnset, CQTStamped

from diagnostic_msgs.msg import DiagnosticStatus, DiagnosticArray
from moveit_msgs.msg import ExecuteTrajectoryActionGoal, ExecuteTrajectoryActionResult, PlanningScene, DisplayTrajectory
from audio_common_msgs.msg import AudioInfo, AudioData, AudioDataStamped
from visualization_msgs.msg import MarkerArray, Marker
from sr_robot_msgs.msg import BiotacAll
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped, Pose, Point
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Bool as BoolMsg, Float32 as Float32Msg, Header, String as StringMsg, Float64MultiArray
from nav_msgs.msg import Path
from dynamic_reconfigure.msg import Config as DynamicReconfigureConfig, ConfigDescription as DynamicReconfigureConfigDescription

import pickle

import re
import sys

def string_link(string):
    return 'guzheng/'+string+'/head'

def finger_link(finger):
    #return 'rh_'+finger+'_biotac_link'
    return f'rh_{finger}_plectrum'

class Aggregator():
    def __init__(self, audio_tactile_delay= 0.0):
        self.mannequin_mode = False

        self.audio_info= None
        self.audio_delay= rospy.Duration(0.0)
        self.audio_drift= rospy.Duration(0)
        self.aggregation_audio_tactile_delay= rospy.Duration(audio_tactile_delay) # TODO: UNUSED as of now. support custom delay in finalize_episode

        self.episode_count= 0
        self.tf= tf2_ros.Buffer(cache_time= rospy.Duration(30))
        self.start_episode()

        self.store= False
        self.episodes= []
        self.subs= []

        self.topics_ignored= [
            '/guzheng/onsets_latest',
            '/guzheng/onsets_markers',
            '/fingertips/plucks_latest',
            '/pluck/keypoint',
            '/pluck/projected_img',
            '/run_episode/goal',
            '/run_episode/result',
        ]

        self.topics= [
            # native PR2
            ('/joint_states', JointState, self.joint_states_cb),
            ('/hand/rh/tactile', BiotacAll, self.biotac_cb),
            ('/tf', TFMessage, self.tf_cb),
            ('/tf_static', TFMessage, self.tf_static_cb),
            ('/diagnostics_agg', DiagnosticArray, self.diagnostics_cb),
            ('/mannequin_mode_active', BoolMsg, self.mannequin_mode_cb),

            # guzheng
            ('/guzheng/audio_stamped', AudioDataStamped, self.audio_cb),
            ('/guzheng/audio_info', AudioInfo, self.audio_info_cb),

            # MoveIt
            ('/move_group/monitored_planning_scene', PlanningScene, self.monitored_planning_scene_cb),
            ('/execute_trajectory/goal', ExecuteTrajectoryActionGoal, self.execute_trajectory_goal_cb),
            ('/execute_trajectory/result', ExecuteTrajectoryActionResult, self.execute_trajectory_result_cb),

            # Experimental control
            # episode parameters
            # ('/run_episode/goal', RunEpisodeActionGoal, self.run_episode_goal_cb),
            # ('/run_episode/result', RunEpisodeActionResult, self.run_episode_result_cb),
            ('/episode/state', EpisodeState, self.state_cb),
            ('/episode/action_parameters', ActionParameters, self.action_parameter_cb),

            ('/pluck/execute_path/goal', ExecutePathActionGoal, self.execute_path_cb),
            ('/pluck/execute_path/result', ExecutePathActionResult, self.execute_path_result_cb),

            ('/pluck/pluck/goal', ExecutePathActionGoal, self.pluck_cb),
            ('/pluck/pluck/result', ExecutePathActionResult, self.pluck_result_cb),
            ('/pluck/commanded_path', Path, self.pluck_commanded_path_cb),
            ('/pluck/planned_path', Path, self.pluck_planned_path_cb),
            ('/pluck/executed_path', Path, self.pluck_executed_path_cb),
            # ('/pluck/projected_img', Image, self.pluck_projected_img_cb),
            ('/pluck/trajectory', DisplayTrajectory, self.pluck_planned_trajectory_cb),
            ('/pluck/executed_trajectory', DisplayTrajectory, self.pluck_executed_trajectory_cb),
            ('/pluck/active_finger', StringMsg, self.active_finger_cb),
            # ('/pluck/keypoint', Marker, self.keypoint_cb),

            ('/fingertips/plucks', TactilePluck, self.fingertip_plucks_cb),
            # ('/fingertips/plucks_latest', MarkerArray, self.fingertip_plucks_latest_cb),
            ('/fingertips/pluck_detector/signal', Float64MultiArray, self.pluck_detector_signal_cb),
            ('/fingertips/pluck_detector/detection', Float64MultiArray, self.pluck_detector_detection_cb),
            ('/fingertips/pluck_detector/parameter_descriptions', DynamicReconfigureConfigDescription, self.pluck_detector_parameter_desc_cb),
            ('/fingertips/pluck_detector/parameter_updates', DynamicReconfigureConfig, self.pluck_detector_parameter_cb),
            ('/fingertips/pluck_projector/parameter_updates', DynamicReconfigureConfig, self.pluck_parameter_cb),
            ('/fingertips/pluck_projector/parameter_descriptions', DynamicReconfigureConfigDescription, self.pluck_parameter_desc_cb),

            ('/guzheng/onsets', NoteOnset, self.onsets_cb),
            # ('/guzheng/onsets_latest', MarkerArray, self.onsets_latest_cb),
            ('/guzheng/cqt', CQTStamped, self.cqt_cb),
            # ('/guzheng/onset_detector/envelope', Float32Msg, self.envelope_cb),
            ('/guzheng/onset_detector/compute_time', Float32Msg, self.compute_time_cb),
            ('/guzheng/onset_detector/drift', Float32Msg, self.drift_cb),
            ('/guzheng/spectrogram', Image, self.spectrogram_cb),
            ('/guzheng/onset_projector/parameter_updates', DynamicReconfigureConfig, self.onset_parameter_cb),
            ('/guzheng/onset_projector/parameter_descriptions', DynamicReconfigureConfigDescription, self.onset_parameter_desc_cb),

            # processed data
            ('/guzheng/fitted_strings', MarkerArray, self.fitted_strings_cb),
            ]

        self.episode_pub= rospy.Publisher('pluck_episodes', PluckEpisodeV2, queue_size= 100, tcp_nodelay= True)

    def live(self):
        if len(self.subs) > 0:
            rospy.logerr('aggregator already subscribed to topics')
            return
        for t,m,c in self.topics:
            self.subs.append(rospy.Subscriber(t, m, c, queue_size= 10, tcp_nodelay= True))

    def bag(self, bag):
        self.store= True

        dispatch= { t : c for t,m,c in self.topics }
        for topic, msg, stamp in bag.read_messages():
            if topic in dispatch:
                dispatch[topic](msg) #, stamp)
            elif topic not in self.topics_ignored:
                rospy.logwarn(f'processed message without callback on topic {topic}')
            if rospy.is_shutdown():
                print('Abort')
                break
        # dumping to pickle is broken because it seems to store messages
        # from this package with a reference to a magic 'tmpxyz' python module
        # and pickle fails trying to load them again
        m= re.match('(.*)\.bag$', bag.filename)
        name= m.group(1) if m else bag.filename
        #pkl_name = name + ".pkl"
        #rospy.loginfo(f'Storing episodes in {pkl_name}')
        #pickle.dump(self.episodes, open(pkl_name, 'wb'))
        episode_bag_name = name+'_extracted_episodes.bag'
        rospy.loginfo(f'write all {len(self.episodes)} episodes to {episode_bag_name}')
        with Bag(episode_bag_name, 'w') as bag:
            for i,e in enumerate(self.episodes):
                rospy.loginfo(f"writing episode {i}")
                bag.write('/pluck_episodes', e, e.header.stamp)
        rospy.loginfo('done')
        self.store= False

    def start_episode(self):
        self.episode= PluckEpisodeV2()
        self.episode.id= None
        self.episode.audio_data.header= None
        self.episode.audio_info= self.audio_info
        self.episode.start_state.is_diff = True
        self.episode.start_state.joint_state.name= []
        self.episode.start_state.joint_state.position= []
        self.episode.start_state.joint_state.velocity= []
        self.episode.start_state.joint_state.effort= []
        self.episode.start_state.multi_dof_joint_state.joint_names= []
        self.episode.start_state.multi_dof_joint_state.transforms= []
        self.episode.detected_tactile_plucks= []
        self.episode.detected_audio_onsets= []
        self.episode.cqt= None

    def finalize_episode(self):
        try:
            self.episode.string_head_frame = self.tf.lookup_transform('base_footprint', string_link(self.episode.string), rospy.Time())
        except tf2_ros.TransformException as e:
            rospy.logwarn(e)
        try:
            finger_tf= self.tf.lookup_transform(string_link(self.episode.string), finger_link(self.episode.finger), self.episode.header.stamp)
            self.episode.finger_start_pose= PoseStamped(header= finger_tf.header, pose= Pose(position= finger_tf.transform.translation, orientation= finger_tf.transform.rotation))
        except tf2_ros.TransformException as e:
            rospy.logwarn(e)

        self.episode.audio_data.header.stamp+= self.audio_delay+self.audio_drift
        self.episode.cqt.header.stamp += self.audio_delay

        for o in self.episode.detected_audio_onsets:
            o.header.stamp += self.audio_delay

        self.publishEpisode()

        self.start_episode()

    def publishEpisode(self):
        if self.mannequin_mode:
            rospy.logwarn("not publishing episode with mannequin mode on")
            return
        if self.episode.id is None:
            rospy.logwarn('not publishing partial episode due to missing data or mannequin')
            return

        self.episode_pub.publish(self.episode)
        if self.store:
            self.episodes.append(self.episode)

    def tracksEpisode(self):
        return self.episode.id is not None

    def state_cb(self, msg):
        if msg.state == 'start' and self.episode.id is not None:
            rospy.logerr('received start of episode {}, but still tracking episode {}'.format(msg.episode, self.episode.id))
        elif self.episode.id is not None and msg.episode != self.episode.id:
            rospy.logerr('received state {}/{} but currently episode {} is still tracked'.format(msg.episode,msg.state,self.episode.id))
        elif msg.state == 'end' and self.episode.id is None:
            rospy.logwarn('received end of episode {}, but not tracking any episode'.format(msg.episode))
            self.start_episode()
            return

        if msg.state == 'start':
            rospy.loginfo(f'{self.episode_count}th episode starts at {msg.header.stamp.to_sec()} with id {msg.episode}')
            self.episode_count+= 1
            self.episode.id= msg.episode
            self.episode.header= msg.header
        elif msg.state == 'end':
            # take start / end times from trajectory execution instead
            #self.episode.length= msg.header.stamp - self.episode.header.stamp
            rospy.loginfo(f"  finalize episode at {msg.header.stamp.to_sec()}")
            self.finalize_episode()

    def execute_trajectory_goal_cb(self, msg):
        # this should always be called twice for moving *towards start* and for *plucking*
        # the second call always overwrites the first one
        self.episode.start_execution = msg.header.stamp
    def execute_trajectory_result_cb(self, msg):
        # this should always be called twice for moving *towards start* and for *plucking*
        # the second call always overwrites the first one
        self.episode.execution_status= msg.result.error_code
        self.episode.length = msg.header.stamp - self.episode.header.stamp


    def joint_states_cb(self, msg):
        if self.tracksEpisode():
            try:
                self.episode.start_state.joint_state.name.index(msg.name[0])
            except ValueError:
                # use time stamp of last part of the joint state
                self.episode.start_state.joint_state.header= msg.header
                self.episode.start_state.joint_state.name.extend(msg.name)
                self.episode.start_state.joint_state.position.extend(msg.position)
                self.episode.start_state.joint_state.velocity.extend(msg.velocity)
                self.episode.start_state.joint_state.effort.extend(msg.effort)
    def tf_cb(self, msg):
        for t in msg.transforms:
            self.tf.set_transform(t, "bag")
    def tf_static_cb(self, msg):
        for t in msg.transforms:
            if t.child_frame_id.endswith("_plectrum"):
                joint_name = t.child_frame_id[len("rh_"):-len("_plectrum")]+"_plectrum_mount"
                if joint_name not in self.episode.start_state.multi_dof_joint_state.joint_names:
                    # add joint with values from t
                    self.episode.start_state.multi_dof_joint_state.joint_names.append(joint_name)
                    self.episode.start_state.multi_dof_joint_state.transforms.append(t.transform)
            self.tf.set_transform_static(t, "bag")
    def onset_parameter_cb(self, msg):
        self.audio_delay = rospy.Duration(next(p.value for p in msg.doubles if p.name == 'delta_t'))
    def pluck_executed_path_cb(self, msg):
        self.episode.executed_path= msg
    def pluck_executed_trajectory_cb(self, msg):
        self.episode.executed_trajectory= msg.trajectory[0].joint_trajectory
    def pluck_planned_path_cb(self, msg):
        self.episode.planned_path= msg
    def pluck_planned_trajectory_cb(self, msg):
        self.episode.planned_trajectory= msg.trajectory[0].joint_trajectory
    def pluck_result_cb(self, msg):
        if not self.tracksEpisode():
            rospy.logwarn(f'got ExecutePath result at {msg.header.stamp.to_sec()}, but no episode is tracked')
    def audio_info_cb(self, msg):
        # persists across whole aggregation
        self.audio_info= msg
        self.episode.audio_info= self.audio_info
    def audio_cb(self, msg):
        if self.tracksEpisode():
            if self.episode.audio_data.header is None:
                self.episode.audio_data.header= msg.header
            self.episode.audio_data.audio.data+= msg.audio.data
    def drift_cb(self, msg):
        self.audio_drift = rospy.Duration(msg.data)
    def spectrogram_cb(self, msg):
        pass
    def cqt_cb(self, msg):
        if not self.tracksEpisode():
            return
        if self.episode.cqt is None:
            self.episode.cqt= msg
        else:
            self.episode.cqt.data+= msg.data
    def onsets_cb(self, msg):
        if not self.tracksEpisode():
            rospy.logwarn(f"found note onset for note '{msg.note}' at {msg.header.stamp.to_sec()} without tracking an episode. Ignoring")
        else:
            self.episode.detected_audio_onsets.append(msg)
    def pluck_cb(self, msg):
        rospy.loginfo(f'  sent path for {msg.goal.finger} in frame {msg.goal.path.header.frame_id} at {msg.header.stamp.to_sec()}')
        #if not self.tracksEpisode():
        #    rospy.logwarn(f"got execute path goal at {msg.header.stamp}, but not tracking an episode")
    def active_finger_cb(self, msg):
        self.episode.finger= msg.data
    def pluck_commanded_path_cb(self, msg):
        self.episode.commanded_path= msg

    def biotac_cb(self, msg):
        fingers= ['ff', 'mf', 'rf', 'lf', 'th']
        if self.tracksEpisode() and self.episode.finger != '' and self.episode.finger not in fingers:
            rospy.logerr(f'unknown finger \'{self.episode.finger}\'')
        if self.tracksEpisode() and self.episode.finger in fingers:
            self.episode.tactile_data.append(BiotacStamped(
                header= msg.header,
                tactile= msg.tactiles[fingers.index(self.episode.finger)]
                ))
            
    def fingertip_plucks_cb(self, msg):
        if not self.tracksEpisode():
            rospy.logwarn(f"found fingertip pluck for finger '{msg.finger}' at {msg.header.stamp.to_sec()} without tracking an episode. Ignoring")
            return
        if self.episode.finger != msg.finger:
            rospy.logwarn(f"found fingertip pluck for finger '{msg.finger}' at {msg.header.stamp.to_sec()} while tracking finger '{self.episode.finger}'. Ignoring")
            return

        self.episode.detected_tactile_plucks.append(msg)

    def action_parameter_cb(self, msg):
        self.episode.action_parameters= msg
        self.episode.string= re.match("guzheng/(.*)/head", msg.header.frame_id).group(1)

    def mannequin_mode_cb(self, msg):
        if msg.data:
            rospy.logerr("Mannequin mode is active. Following episode recording is invalid.")
            self.mannequin_mode = True
        else:
            if self.mannequin_mode:
                rospy.loginfo("Mannequin mode is inactive again. Recording is valid again from here on.")
            self.mannequin_mode = False
        
    def diagnostics_cb(self, msg):
        for status in msg.status:
            if status.level >= DiagnosticStatus.ERROR:
                rospy.logerr('diagnostics at {}: {}: {}'.format(msg.header.stamp, status.name, status.message))

    def pluck_parameter_cb(self, msg):
        pass
    def pluck_parameter_desc_cb(self, msg):
        pass
    def fitted_strings_cb(self, msg):
        pass
    def onset_parameter_desc_cb(self, msg):
        pass
    def pluck_detector_signal_cb(self, msg):
        pass
    def pluck_detector_detection_cb(self, msg):
        pass
    def pluck_detector_parameter_cb(self, msg):
        pass
    def pluck_detector_parameter_desc_cb(self, msg):
        pass
    def plucks_latest_cb(self, msg):
        pass
    def execute_path_result_cb(self, msg):
        pass
    def execute_path_cb(self, msg):
        pass
    def fingertip_plucks_latest_cb(self, msg):
        pass
    def monitored_planning_scene_cb(self, msg):
        pass
    def compute_time_cb(self, msg):
        pass


if __name__ == '__main__':
    rospy.init_node('aggregator')
    if len(sys.argv) < 2:
        print("usage: {filename.bag|live}")
        sys.exit(1)
    if sys.argv[1] == 'live':
        Aggregator().live()
        rospy.spin()
    else:
        Aggregator().bag(Bag(sys.argv[1]))
        rospy.signal_shutdown("")
