#!/usr/bin/env python

from std_msgs.msg import String as StringMsg
from tams_pr2_guzheng.msg import PlayPieceAction
from tams_pr2_guzheng.msg import PlayPieceGoal
from moveit_commander import MoveGroupCommander
from music_perception.msg import NoteOnset, Piece

import rospy
import actionlib

class RepeatAfterMe:
    SILENCE_BEFORE_DECIDING_TO_REPLAY = rospy.Duration(3.0)
    LONG_PIECE_THRESHOLD = 5
    CLUSTER_THRESHOLD = 0.3

    def __init__(self):
        self.piece= Piece()
        self.recording= True
        self.onsets_sub= rospy.Subscriber("guzheng/onsets", NoteOnset, self.onset_cb)
        self.play_piece= actionlib.SimpleActionClient("play_piece/action", PlayPieceAction)
        self.say = rospy.Publisher("/say", StringMsg, queue_size=1)
        self.move_group = MoveGroupCommander("manipulation")
        self.move_group.set_max_velocity_scaling_factor(0.7)
        self.move_group.set_max_acceleration_scaling_factor(0.7)
        self.play_piece.wait_for_server()
        rospy.sleep(1.0) # buffer time for /say publisher

    def go_home(self):
        self.move_group.set_named_target("guzheng_initial")
        self.move_group.go(wait=True)

    @staticmethod
    def clean_piece(piece):
        cleaned= Piece()
        cleaned.header= piece.header

        # only take the last onset if multiple onsets are close together
        last_onset= None
        for onset in piece.onsets:
            if last_onset is None:
                last_onset= onset
            elif onset.header.stamp - last_onset.header.stamp > rospy.Duration(RepeatAfterMe.CLUSTER_THRESHOLD):
                cleaned.onsets.append(last_onset)
                last_onset= onset
            else:
                last_onset= onset
        if last_onset is not None:
            cleaned.onsets.append(last_onset)

        return cleaned

    def spin(self):
        first_time= True
        self.go_home()
        self.say.publish("I'm listening. Please show me your talent.")
        while True:
            if not first_time:
                self.say.publish("It's your turn again.")
            first_time= False
            while (
                not rospy.is_shutdown()
                and self.recording
                and (len(self.piece.onsets) == 0 
                    or rospy.Time.now() - self.piece.onsets[-1].header.stamp < self.SILENCE_BEFORE_DECIDING_TO_REPLAY)):
                rospy.sleep(0.2)

            if rospy.is_shutdown():
                return

            rospy.loginfo("Repeating sequence.")
            self.recording= False

            goal = PlayPieceGoal()
            goal.piece = self.clean_piece(self.piece)
            self.piece = Piece()

            if len(goal.piece.onsets) < self.LONG_PIECE_THRESHOLD:
                self.say.publish("I can play that too. Let me try.")
                rospy.sleep(2.0)
            else:
                self.say.publish("I think I can play that too. But I'll have to think about it for a moment.")
            
            self.play_piece.send_goal(goal)
            self.play_piece.wait_for_result()
            self.go_home()
            self.recording= True

    def onset_cb(self, onset : NoteOnset):
        rospy.loginfo(f"got onset with note {onset.note}")
        if self.recording == False:
            return

        if onset.note != '':
            if len(self.piece.onsets) == 0:
                self.piece.header = onset.header
            self.piece.onsets.append(onset)

def main():
    rospy.init_node("piece_recorder")
    p = RepeatAfterMe()
    p.spin()

if __name__ == "__main__":
    main()
