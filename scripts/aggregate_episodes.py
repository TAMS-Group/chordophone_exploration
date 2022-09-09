#!/usr/bin/env python

from rosbag import Bag
import rospy
import tf2_ros

from tams_pr2_guzheng.msg import (
    PluckEpisodeV1,
    BiotacStamped,
    EpisodeState,
    ActionParameters,
    NoteOnset,
    CQTStamped,
    ExecutePathActionGoal,
    ExecutePathActionResult,
    )

from diagnostic_msgs.msg import DiagnosticStatus, DiagnosticArray
from moveit_msgs.msg import ExecuteTrajectoryActionGoal, ExecuteTrajectoryActionResult, PlanningScene, DisplayTrajectory
from audio_common_msgs.msg import AudioInfo, AudioData, AudioDataStamped
from visualization_msgs.msg import MarkerArray
from sr_robot_msgs.msg import BiotacAll
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Pose, Point
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Bool as BoolMsg, Float32 as Float32Msg, Header
from nav_msgs.msg import Path
from dynamic_reconfigure.msg import Config as DynamicReconfigureConfig

import pickle

import re
import sys

def string_link(note):
    return 'guzheng/'+note+'/head'

def finger_link(finger):
    return 'rh_'+finger+'_biotac_link'

class Aggregator():
    def __init__(self):
        self.audio_info= None
        self.audio_delay= None
        self.finger_tip_offset= Point()

        self.episode_count= 0
        self.tf= tf2_ros.Buffer(cache_time= rospy.Duration(30))
        self.start_episode()

        self.store= False
        self.episodes= []
        self.subs= []

        self.topics= [
            # monitoring
            ('/diagnostics_agg', DiagnosticArray, self.diagnostics_cb),
            ('/mannequin_mode_active', BoolMsg, self.mannequin_mode_cb),
            ('/move_group/monitored_planning_scene', PlanningScene, self.monitored_planning_scene_cb),
            ('/guzheng/onset_detector/compute_time', Float32Msg, self.compute_time_cb),

            ('/guzheng/onset_projector/parameter_updates', DynamicReconfigureConfig, self.onset_parameter_cb),
            ('/guzheng/pluck_projector/parameter_updates', DynamicReconfigureConfig, self.pluck_parameter_cb),

            # episode parameters
            ('/episode/state', EpisodeState, self.state_cb),
            ('/episode/action_parameters', ActionParameters, self.action_parameter_cb),
            ('/pluck/execute_path/goal', ExecutePathActionGoal, self.execute_path_cb),
            ('/pluck/execute_path/result', ExecutePathActionResult, self.pluck_execute_path_result_cb),
            ('/pluck/planned_path', Path, self.pluck_planned_path_cb),
            ('/pluck/trajectory', DisplayTrajectory, self.pluck_planned_trajectory_cb),

            # execution monitoring
            ('/pluck/executed_path', Path, self.pluck_executed_path_cb),
            ('/pluck/executed_trajectory', DisplayTrajectory, self.pluck_executed_trajectory_cb),

            # actual execution
            ('/execute_trajectory/goal', ExecuteTrajectoryActionGoal, self.execute_trajectory_goal_cb),
            ('/execute_trajectory/result', ExecuteTrajectoryActionResult, self.execute_trajectory_result_cb),

            # sensor events
            ('/guzheng/onsets', NoteOnset, self.onsets_cb),
            ('/guzheng/plucks', MarkerArray, self.plucks_cb),

            ('/guzheng/audio', AudioData, self.audio_cb),
#            ('/guzheng/audio_stamped', AudioDataStamped, self.audio_cb),
            ('/guzheng/audio_info', AudioInfo, self.audio_info_cb),
            ('/guzheng/cqt', CQTStamped, self.cqt_cb),

            # streams
            ('/hand/rh/tactile', BiotacAll, self.biotac_cb),
            ('/joint_states', JointState, self.joint_states_cb),
            ('/tf', TFMessage, self.tf_cb),
            ('/tf_static', TFMessage, self.tf_static_cb),
            ]

        self.episode_pub= rospy.Publisher('pluck_episodes', PluckEpisodeV1, queue_size= 100, tcp_nodelay= True)

    def live(self):
        if len(self.subs) > 0:
            rospy.logerror('aggregator already subscribed to topics')
            return
        for t,m,c in self.topics:
            self.subs.append(rospy.Subscriber(t, m, c, queue_size= 10))

    def bag(self, bag):
        self.store= True

        dispatch= { t : c for t,m,c in self.topics }
        for topic, msg, stamp in bag.read_messages():
            if topic in dispatch:
                dispatch[topic](msg) #, stamp)
            else:
                rospy.logwarn(f'processed message without callback on topic {topic}')
            if rospy.is_shutdown():
                print('Abort')
                break
        ## broken because it seems to store messages from this package with a reference
        ## to a magic 'tmpxyz' python module and pickle fails trying to load them again
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
        self.episode= PluckEpisodeV1()
        self.episode.id= None
        self.episode.audio_data.header= None
        self.episode.audio_info= self.audio_info
        self.episode.audio_delay= self.audio_delay
        self.episode.start_state.name= []
        self.episode.start_state.position= []
        self.episode.start_state.velocity= []
        self.episode.start_state.effort= []
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

        if self.episode.audio_data.header is None:
            self.episode.audio_data.header = Header()
        if self.episode.id is not None:
            self.episode_pub.publish(self.episode)
            if self.store:
                self.episodes.append(self.episode)
        else:
            rospy.logwarn('not publishing partial episode due to missing data')

        self.start_episode()

    def tracksEpisode(self):
        return self.episode.id is not None

    def joint_states_cb(self, msg):
        if self.tracksEpisode():
            try:
                self.episode.start_state.name.index(msg.name[0])
            except ValueError:
                # use time stamp of last part of the joint state
                self.episode.start_state.header= msg.header
                self.episode.start_state.name.extend(msg.name)
                self.episode.start_state.position.extend(msg.position)
                self.episode.start_state.velocity.extend(msg.velocity)
                self.episode.start_state.effort.extend(msg.effort)
    def tf_cb(self, msg):
        for t in msg.transforms:
            self.tf.set_transform(t, "bag")
    def tf_static_cb(self, msg):
        for t in msg.transforms:
            self.tf.set_transform_static(t, "bag")
    def pluck_parameter_cb(self, msg):
        pass

    def onset_parameter_cb(self, msg):
        self.audio_delay = rospy.Duration(next(p.value for p in msg.doubles if p.name == 'delta_t'))
        self.finger_tip_offset.x = next(p.value for p in msg.doubles if p.name == 'offset_x')
        self.finger_tip_offset.y = next(p.value for p in msg.doubles if p.name == 'offset_y')
        self.finger_tip_offset.z = next(p.value for p in msg.doubles if p.name == 'offset_z')

    def pluck_executed_path_cb(self, msg):
        if not self.tracksEpisode():
            rospy.logwarn("received executed path, but not tracking an episode")
        self.episode.executed_path= msg
    def pluck_executed_trajectory_cb(self, msg):
        if not self.tracksEpisode():
            rospy.logwarn("received executed trajectory, but not tracking an episode")
        self.episode.executed_trajectory= msg.trajectory[0].joint_trajectory
    def pluck_planned_path_cb(self, msg):
        #if not self.tracksEpisode():
        #    rospy.logwarn("received planned path, but not tracking an episode")
        self.episode.planned_path= msg
    def pluck_planned_trajectory_cb(self, msg):
        #if not self.tracksEpisode():
        #    rospy.logwarn("received planned trajectory, but not tracking an episode")
        self.episode.planned_trajectory= msg.trajectory[0].joint_trajectory
    def pluck_execute_path_result_cb(self, msg):
        if not self.tracksEpisode():
            rospy.logwarn(f'got ExecutePath result at {msg.header.stamp.to_sec()}, but no episode is tracked')
        self.episode.planned_path = msg.result.generated_path
        self.episode.executed_path = msg.result.executed_path
        self.episode.planned_trajectory = msg.result.generated_trajectory
        self.episode.executed_trajectory = msg.result.executed_trajectory
        pass
    def audio_info_cb(self, msg):
        # persists across whole aggregation
        self.audio_info= msg
        self.episode.audio_info= self.audio_info
    def audio_cb(self, msg):
        if self.tracksEpisode():
            if self.episode.audio_data.header is None and hasattr(msg, "header"):
                self.episode.audio_data.header= msg.header
            self.episode.audio_data.data+= msg.data
    def cqt_cb(self, msg):
        if not self.tracksEpisode():
            return
        # TODO: check whether msg.header.stamp matches to continue previous message
        if self.episode.cqt is None:
            self.episode.cqt= msg
        else:
            self.episode.cqt.data+= msg.data
    def onsets_cb(self, msg):
        if not self.tracksEpisode():
            rospy.logwarn(f"found note onset for note '{msg.note}' at {msg.header.stamp.to_sec()} without tracking an episode. Ignoring")
        else:
            self.episode.detected_audio_onsets.append(msg)
    def plucks_cb(self, msg):
        if not self.tracksEpisode():
            rospy.logwarn(f"found pluck event at {msg.markers[0].header.stamp.to_sec()} without tracking an episode. Ignoring")
        elif len(msg.markers) == 0:
            rospy.logerr("found empty plucks marker message")
        else:
            self.episode.detected_tactile_plucks.append(msg.markers[0].header.stamp)
    def execute_path_cb(self, msg):
        rospy.loginfo(f'  sent path for {msg.goal.finger} in frame {msg.goal.path.header.frame_id} at {msg.header.stamp.to_sec()}')
        #if not self.tracksEpisode():
        #    rospy.logwarn(f"got execute path goal at {msg.header.stamp}, but not tracking an episode")
        self.episode.string= re.match("guzheng/(.*)/head", msg.goal.path.header.frame_id).group(1)
        self.episode.commanded_path= msg.goal.path
        self.episode.finger= msg.goal.finger
        self.episode.calibrated_tip= self.finger_tip_offset
    def biotac_cb(self, msg):
        fingers= ['ff', 'mf', 'rf', 'lf', 'th']
        if self.tracksEpisode() and self.episode.finger != '' and self.episode.finger not in fingers:
            rospy.logerr(f'unknown finger \'{self.episode.finger}\'')
        if self.tracksEpisode() and self.episode.finger in fingers:
            self.episode.tactile_data.append(BiotacStamped(
                header= msg.header,
                tactile= msg.tactiles[fingers.index(self.episode.finger)]
                ))
    def state_cb(self, msg):
        if self.episode.id is not None and msg.episode != self.episode.id:
            rospy.logerr('received state {}/{} but currently episode {} is still tracked'.format(msg.episode,msg.state,self.episode.id))
        if msg.state == 'start' and self.episode.id is not None:
            rospy.logerr('received start of episode {}, but still tracking episode {}'.format(msg.episode, self.episode.id))

        if msg.state == 'end':
            # take start / end times from trajectory execution instead
            #self.episode.length= msg.header.stamp - self.episode.header.stamp
            rospy.loginfo(f"  finalize episode at {msg.header.stamp.to_sec()}")
            self.finalize_episode()
        if msg.state == 'start':
            rospy.loginfo(f'{self.episode_count}th episode starts at {msg.header.stamp.to_sec()} with id {msg.episode}')
            self.episode_count+= 1
            self.episode.id= msg.episode
            self.episode.header= msg.header
    def action_parameter_cb(self, msg):
        self.episode.action_parameters= msg
    def execute_trajectory_goal_cb(self, msg):
        # this should always be called twice for moving *towards start* and for *plucking*
        # the second call always overwrites the first one
        self.episode.start_execution = msg.header.stamp
    def execute_trajectory_result_cb(self, msg):
        # this should always be called twice for moving *towards start* and for *plucking*
        # the second call always overwrites the first one
        self.episode.execution_status= msg.result.error_code
        self.episode.length = msg.header.stamp - self.episode.header.stamp

    def monitored_planning_scene_cb(self, msg):
        pass
    def compute_time_cb(self, msg):
        pass
    def mannequin_mode_cb(self, msg):
        if msg.data:
            rospy.logerr("Mannequin mode is active. Recording is invalid.")
    def diagnostics_cb(self, msg):
        for status in msg.status:
            if status.level >= DiagnosticStatus.ERROR:
                rospy.logerr('diagnostics at {}: {}: {}'.format(msg.header.stamp, status.name, status.message))


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
