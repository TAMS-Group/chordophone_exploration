#!/usr/bin/env python

from tams_pr2_guzheng.onset_to_path import OnsetToPath
from tams_pr2_guzheng.msg import (
    RunEpisodeAction,
    RunEpisodeGoal,
    RunEpisodeActionResult,
    RunEpisodeResult,
    ExecutePathAction,
    ExecutePathGoal,
    PlayPieceAction
)
from music_perception.msg import NoteOnset, Piece
from tams_pr2_guzheng.utils import row_from_result, stitch_paths, run_params

from tams_pr2_guzheng.paths import RuckigPath

import librosa

import copy
import rospy
import tf2_ros
import tf2_geometry_msgs
import actionlib
import rospkg

class PlayPiece:
    def __init__(self):
        self.tf= tf2_ros.Buffer()
        self.tf_listener= tf2_ros.TransformListener(self.tf)

        # self.run_episode = actionlib.SimpleActionClient('run_episode', RunEpisodeAction)
        # self.run_episode.wait_for_server()

        self.execute_path = actionlib.SimpleActionClient('pluck/execute_path', ExecutePathAction)

        self.o2p= OnsetToPath(storage= rospy.get_param("~storage", rospkg.RosPack().get_path("tams_pr2_guzheng") + "/data/plucks.json"))
        self.start_string_position = rospy.get_param("~start_string_position", None)

        self.run_episode_result_sub= rospy.Subscriber('run_episode/result', RunEpisodeActionResult, self.run_episode_result_cb)

        self.piece_sub= rospy.Subscriber('piece', Piece, self.piece_cb)
        self.piece_sub= rospy.Subscriber('piece_midi_loudness', Piece, self.piece_midi_loudness_cb)

        self.piece_action= actionlib.SimpleActionServer('play_piece', PlayPieceAction, self.play_piece_cb, auto_start=False)
        self.piece_action.start()

    def run_episode_result_cb(self, msg):
        if len(msg.result.onsets) > 0:
            rospy.loginfo(f"add pluck with perceived note '{msg.result.onsets[-1].note}' ({msg.result.onsets[-1].loudness:.2F}dB) to table")
            self.o2p.add_sample(row_from_result(msg.result))
            if len(self.o2p.pluck_table) % 10 == 1:
                self.o2p.print_summary()
        else:
            rospy.loginfo(f'ignoring result with no detected onsets')

    def play_piece_cb(self, goal):
        self.piece_cb(goal.piece)
        self.piece_action.set_succeeded()

    def piece_midi_loudness_cb(self, msg):
        # loudness of msg.onsets is 1-127, we scale it between min and max in o2p.pluck_table

        for o in msg.onsets:
            loudness_min, loudness_max = self.o2p.get_note_min_max(o.note)
            loudness_range = loudness_max - loudness_min
            o.loudness = loudness_min + loudness_range * (o.loudness-1) / 127.0

        self.piece_cb(msg)

    def piece_cb(self, msg):
        rospy.loginfo(f"get to play piece with {len(msg.onsets)} onsets")
        if not self.execute_path.wait_for_server(timeout= rospy.Duration(5.0)):
            rospy.logerr("execute_path action is not connected. Cannot execute motions to play.")
            return
        paths= []
        last_midi_note = None
        direction= -1.0
        last_string_position = self.start_string_position
        finger = None
        for o in msg.onsets:
            midi_note= librosa.note_to_midi(o.note)
            if last_midi_note is not None:
                if midi_note > last_midi_note:
                    direction= 1.0
                elif midi_note < last_midi_note:
                    direction= -1.0
                # elif midi_note == last_midi_note:
                #     direction*= -1.0

            last_midi_note= midi_note
            try:
                # TODO: We can't mix fingers here because ExecutePath only supports one finger in request.
                path, finger, prob = self.o2p.get_path(note=o.note, loudness= o.loudness, direction= direction, string_position= last_string_position, finger= finger)
                path = path()
                last_string_position = path.poses[0].pose.position.x
            except ValueError as e:
                rospy.logerr(f"No known way to play note {o.note}, will skip it. ({e})")
                continue
            approach_path = copy.deepcopy(path)
            approach_path.poses = approach_path.poses[0:1]
            approach_pose = copy.deepcopy(approach_path.poses[0])
            approach_pose.pose.position.z += 0.010
            approach_path.poses.insert(0, approach_pose)
            paths.append(approach_path)
            paths.append(path)

        try:
            stitched_path = stitch_paths(paths, self.tf)
        except tf2_ros.TransformException as e:
            rospy.logerr("will not attempt execution")
            return

        stitched_path.poses = stitched_path.poses[::3]
        self.execute_path.send_goal(ExecutePathGoal(path= stitched_path, finger= finger))
        self.execute_path.wait_for_result()


if __name__ == '__main__':
    rospy.init_node('play_piece')

    PlayPiece()
    rospy.spin()
