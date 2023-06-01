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
        self.execute_path.wait_for_server()

        self.o2p= OnsetToPath(rospy.get_param("~storage", rospkg.RosPack().get_path("tams_pr2_guzheng") + "/data/plucks.json"))
        self.print_summary()

        self.run_episode_result_sub= rospy.Subscriber('run_episode/result', RunEpisodeActionResult, self.run_episode_result_cb)

        rospy.on_shutdown(self.store_plucks)

        self.piece_sub= rospy.Subscriber('piece', Piece, self.piece_cb)
        self.piece_action= actionlib.SimpleActionServer('play_piece', PlayPieceAction, self.play_piece_cb, auto_start=False)
        self.piece_action.start()

    def print_summary(self):
        summary= f"OnsetToPath stores {len(self.o2p.pluck_table)} plucks\n"
        for n in set(self.o2p.pluck_table['detected_note']):
            summary+= f"{n}: {len(self.o2p.pluck_table[self.o2p.pluck_table['detected_note'] == n])} plucks\n"
        rospy.loginfo(summary)


    def run_episode_result_cb(self, msg):
        if len(msg.result.onsets) > 0:
            rospy.loginfo(f"add pluck with perceived note '{msg.result.onsets[-1].note}' ({msg.result.onsets[-1].loudness:.2F}dB) to table")
            self.o2p.add_sample(row_from_result(msg.result))
            if len(self.o2p.pluck_table) % 10 == 1:
                self.print_summary()
        else:
            rospy.loginfo(f'ignoring result with no detected onsets')

    def store_plucks(self):
        rospy.loginfo(f"storing plucks in {self.o2p.storage}")
        self.o2p.store_to_file()

    def play_piece_cb(self, goal):
        self.piece_cb(goal.piece)
        self.piece_action.set_succeeded()

    def piece_cb(self, msg):
        rospy.loginfo(f"playing piece with {len(msg.onsets)} onsets")
        paths= []
        last_midi_note = 0
        direction= 1.0
        last_string_position = 0.1
        for o in msg.onsets:
            midi_note= librosa.note_to_midi(o.note)
            if midi_note > last_midi_note:
                direction= 1.0
            elif midi_note < last_midi_note:
                direction= -1.0
            elif midi_note == last_midi_note:
                direction*= -1.0

            last_midi_note= midi_note
            try:
                p = self.o2p.get_path(note=o.note, loudness= o.loudness, direction= direction, string_position= last_string_position)
                last_string_position = p.poses[0].pose.position.x
            except ValueError:
                rospy.logerr(f"No known way to play note {o.note}, will skip it.")
                continue
            approach_path = copy.deepcopy(p)
            approach_path.poses = approach_path.poses[0:1]
            approach_pose = copy.deepcopy(approach_path.poses[0])
            approach_pose.pose.position.z += 0.010
            approach_path.poses.insert(0, approach_pose)
            paths.append(approach_path)
            paths.append(p)

        try:
            stitched_path = stitch_paths(paths, self.tf)
        except tf2_ros.TransformException as e:
            rospy.logerr("will not attempt execution")
            return
        self.execute_path.send_goal(ExecutePathGoal(path= stitched_path, finger= 'ff'))
        self.execute_path.wait_for_result()


if __name__ == '__main__':
    rospy.init_node('play_piece')

    play_piece= PlayPiece()
    rospy.spin()
