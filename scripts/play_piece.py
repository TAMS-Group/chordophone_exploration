#!/usr/bin/env python

from tams_pr2_guzheng.onset_to_path import OnsetToPath
from tams_pr2_guzheng.msg import (
    RunEpisodeAction,
    RunEpisodeGoal,
    RunEpisodeActionResult,
    RunEpisodeResult,
    ExecutePathAction,
    ExecutePathGoal
)
from music_perception.msg import NoteOnset, Piece
from tams_pr2_guzheng.utils import row_from_result, stitch_paths, run_params

from tams_pr2_guzheng.paths import RuckigPath

import copy

import rospy
import tf2_ros
import tf2_geometry_msgs
import actionlib

class PlayPiece:
    def __init__(self):
        self.tf= tf2_ros.Buffer()
        self.tf_listener= tf2_ros.TransformListener(self.tf)

        # self.run_episode = actionlib.SimpleActionClient('run_episode', RunEpisodeAction)
        # self.run_episode.wait_for_server()

        self.execute_path = actionlib.SimpleActionClient('pluck/execute_path', ExecutePathAction)
        self.execute_path.wait_for_server()

        self.o2p= OnsetToPath()
        rospy.loginfo(f"OnsetToPath stores {len(self.o2p.pluck_table)} plucks")

        def resultCb(msg):
            if len(msg.result.onsets) > 0:
                rospy.loginfo(f"add pluck with perceived note '{msg.result.onsets[-1].note}' to table")
                self.o2p.add_sample(row_from_result(msg.result))
            else:
                rospy.loginfo(f'ignoring result with no detected onsets')
        self.run_episode_result_sub= rospy.Subscriber('run_episode/result', RunEpisodeActionResult, resultCb)

        self.piece_sub= rospy.Subscriber('piece', Piece, self.pieceCb)

    def pieceCb(self, msg):
        paths= []
        for o in msg.onsets:
            try:
                p = self.o2p.get_path(note=o.note, loudness= o.loudness)
            except Exception:
                rospy.logerr(f"No known way to play note {o.note}, will skip it.")
                continue
            approach_path = copy.deepcopy(p)
            approach_path.poses = approach_path.poses[0:1]
            approach_pose = copy.deepcopy(approach_path.poses[0])
            approach_pose.pose.position.z += 0.010
            approach_path.poses.insert(0, approach_pose)
            paths.append(approach_path)
            paths.append(p)

        stitched_path = stitch_paths(paths, self.tf)
        self.execute_path.send_goal(ExecutePathGoal(path= stitched_path, finger= 'ff'))
        self.execute_path.wait_for_result()
if __name__ == '__main__':
    rospy.init_node('play_piece')

    play_piece= PlayPiece()
    rospy.spin()