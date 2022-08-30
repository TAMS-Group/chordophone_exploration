#!/usr/bin/env python

import rospy
from rosbag import Bag

from tams_pr2_guzheng.msg import PluckEpisodeV1, BiotacStamped, EpisodeState, ActionParameters, NoteOnset, ExecutePathActionGoal, ExecutePathActionResult

from diagnostic_msgs.msg import DiagnosticStatus, DiagnosticArray
from moveit_msgs.msg import ExecuteTrajectoryActionGoal, ExecuteTrajectoryActionResult, PlanningScene, DisplayTrajectory
from audio_common_msgs.msg import AudioInfo, AudioData
from visualization_msgs.msg import MarkerArray
from sr_robot_msgs.msg import BiotacAll
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Bool as BoolMsg
from nav_msgs.msg import Path

import pickle

import re
import sys

class Aggregator():
    def __init__(self):
        self.audio_info= None
        self.start_episode()

        self.store= False
        self.episodes= []
        self.subs= []

        self.topics= [
            # monitoring
            ('/diagnostics_agg', DiagnosticArray, self.diagnostics_cb),
            ('/mannequin_mode_active', BoolMsg, self.mannequin_mode_cb),
            ('/move_group/monitored_planning_scene', PlanningScene, self.monitored_planning_scene_cb),

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
            ('/guzheng/audio_info', AudioInfo, self.audio_info_cb),

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
        rospy.loginfo(f'write all episodes to {episode_bag_name}')
        with Bag(episode_bag_name, 'w') as bag:
            for e in self.episodes:
                bag.write('/pluck_episodes', e, e.header.stamp)
        self.store= False

    def start_episode(self):
        self.episode= PluckEpisodeV1()
        self.episode.id= None
        self.episode.audio_info= self.audio_info
        self.episode.start_state.name= []
        self.episode.start_state.position= []
        self.episode.start_state.velocity= []
        self.episode.start_state.effort= []


    def finalize_episode(self):
        if self.episode.id is not None:
            self.episode_pub.publish(self.episode)
            if self.store:
                self.episodes.append(self.episode)
        else:
            rospy.loginfo('not publishing partial episode due to missing data')

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
        pass
    def tf_cb(self, msg):
        pass
    def tf_static_cb(self, msg):
        pass
    def monitored_planning_scene_cb(self, msg):
        pass
    def pluck_execute_path_result_cb(self, msg):
        pass
    def mannequin_mode_cb(self, msg):
        if msg.data:
            rospy.logerr("Mannequin mode is active. Recording is invalid.")

    def pluck_executed_path_cb(self, msg):
        if not self.tracksEpisode():
            rospy.logwarn("received executed path, but not tracking an episode")
        self.episode.executed_path= msg
    def pluck_executed_trajectory_cb(self, msg):
        if not self.tracksEpisode():
            rospy.logwarn("received executed trajectory, but not tracking an episode")
        self.episode.executed_trajectory= msg.trajectory[0].joint_trajectory
    def pluck_planned_path_cb(self, msg):
        if not self.tracksEpisode():
            rospy.logwarn("received planned path, but not tracking an episode")
        self.episode.planned_path= msg
    def pluck_planned_trajectory_cb(self, msg):
        if not self.tracksEpisode():
            rospy.logwarn("received planned trajectory, but not tracking an episode")
        self.episode.planned_trajectory= msg.trajectory[0].joint_trajectory

    def audio_info_cb(self, msg):
        # persists across whole aggregation
        self.audio_info= msg
        self.episode.audio_info= self.audio_info
    def audio_cb(self, msg):
        if self.tracksEpisode():
            self.episode.audio_data.data+= msg.data
    def onsets_cb(self, msg):
        if not self.tracksEpisode():
            rospy.logwarn(f"found note onset for note '{msg.note}' at {msg.header.stamp} without tracking an episode. Ignoring")
        else:
            self.episode.detected_audio_onsets.append(msg)
    def plucks_cb(self, msg):
        if not self.tracksEpisode():
            rospy.logwarn(f"found pluck event at {msg.header.stamp} without tracking an episode. Ignoring")
        elif len(msg.markers) == 0:
            rospy.logerr("found empty plucks marker message")
        else:
            self.episode.detected_tactile_plucks.append(msg.markers[0].header.stamp)
    def execute_path_cb(self, msg):
        rospy.loginfo(f'execute path for {msg.goal.finger} in frame {msg.goal.path.header.frame_id} at {msg.header.stamp.to_sec()}')
        if not self.tracksEpisode():
            rospy.logwarn(f"got execute path goal at {msg.header.stamp}, but not tracking an episode")
        self.episode.string= re.match("guzheng/(.*)/head", msg.goal.path.header.frame_id).group(1)
        self.episode.commanded_path= msg.goal.path
        self.episode.finger= msg.goal.finger
    def biotac_cb(self, msg):
        fingers= ['ff', 'mf', 'rf', 'lf', 'th']
        if self.tracksEpisode() and self.episode.finger != '' and self.episode.finger not in fingers:
            rospy.logerr(f'unknown finger \'{self.episode.finger}\'')
        if self.tracksEpisode() and self.episode.finger in fingers:
            self.episode.tactile_data.append(BiotacStamped(
                header= msg.header,
                tactile= msg.tactiles[fingers.index(self.episode.finger)]
                ))
    def diagnostics_cb(self, msg):
        for status in msg.status:
            if status.level >= DiagnosticStatus.ERROR:
                rospy.logerr('Diagnostics error at {}:\n{}: {}'.format(msg.header.stamp, status.name, status.message))
    def state_cb(self, msg):
        if self.episode.id is not None and msg.episode != self.episode.id:
            rospy.logerr('received state {}/{} but currently episode {} is still tracked'.format(msg.episode,msg.state,self.episode.id))
        if msg.state == 'start' and self.episode.id is not None:
            rospy.logerr('received start of episode {}, but still tracking episode {}'.format(msg.episode, self.episode.id))

        if msg.state == 'end':
            self.episode.length= msg.header.stamp - self.episode.header.stamp
            rospy.loginfo(f"finalize episode {self.episode.id} at {msg.header.stamp.to_sec()}")
            self.finalize_episode()
        if msg.state == 'start':
            rospy.loginfo(f'episode starts at {msg.header.stamp.to_sec()} with id {msg.episode}')
            self.episode.id= msg.episode
            self.episode.header= msg.header
    def action_parameter_cb(self, msg):
        if not self.tracksEpisode():
            rospy.logwarn('got action parameters, but not actively tracking an episode right now')
        self.episode.action_parameters= msg
    def execute_trajectory_goal_cb(self, msg):
        if not self.tracksEpisode():
            rospy.logwarn('got trajectory goal, but not actively tracking an episode right now')
    def execute_trajectory_result_cb(self, msg):
        if not self.tracksEpisode():
            rospy.logwarn('got trajectory goal, but not actively tracking an episode right now')
        self.episode.execution_status= msg.result.error_code


if __name__ == '__main__':
    rospy.init_node('aggregator')
    if sys.argv[1] == 'live':
        Aggregator.live()
        rospy.spin()
    else:
        Aggregator().bag(Bag(sys.argv[1]))
