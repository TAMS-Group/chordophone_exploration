#!/usr/bin/env python

import rospy
from audio_common_msgs.msg import AudioData, AudioInfo
from geometry_msgs.msg import PointStamped

import numpy as np
import librosa
import struct
import sys
import time

class OnsetDetector():
	buffer= np.array([], dtype= float)

	@staticmethod
	def unpack_data(data):
		return np.array(struct.unpack('{0}h'.format(int(len(data)/2)), bytes(data)), dtype= float)

	def check_audio_format(self):
		rospy.loginfo('Waiting for Audio Info')
		info= rospy.wait_for_message('audio_info', AudioInfo)
		if info.channels != 1:
			rospy.logfatal('audio data has more than one channel, expecting single-channel recording')
		elif info.sample_rate != 44100:
			rospy.logfatal('sample rate {0} is not 44100'.format(info.sample_rate))
		elif info.sample_format != 'S16LE':
			rospy.logfatal('sample format {0} is not S16LE'.format(info.sample_rate))
		elif info.coding_format != 'wave':
			rospy.logfatal('coding "{0}" is not raw'.format(info.coding_format))
		else:
			rospy.loginfo('Audio compatible')
			return True
		return False

	def __init__(self):
		self.sr= 44100
		self.hop_length= 512

		# number of samples for analysis window and overlap regions between consecutive windows
		self.window_t= 3.0
		self.window_overlap_t= 0.5

		self.window= int(self.sr*self.window_t)
		self.window_overlap= int(self.sr*self.window_overlap_t)

		self.buffer_time= None

		if not self.check_audio_format():
			rospy.signal_shutdown('incompatible audio format')
			return

		self.pub= rospy.Publisher('onsets', PointStamped, queue_size= 100)
		self.sub= rospy.Subscriber('audio', AudioData, self.audio_cb, queue_size= 100)

		self.last_time = rospy.Time.now()

	def reset(self):
		self.buffer_time= None
		self.buffer= self.buffer[0:0]

	def audio_cb(self, msg):
		# handle bag loop graciously
		now = rospy.Time.now()
		if now < self.last_time:
			rospy.loginfo('detected bag loop')
			self.reset()

		if self.buffer_time is None:
			self.buffer_time= now

		self.buffer= np.concatenate([self.buffer, OnsetDetector.unpack_data(msg.data)])

		if self.buffer.shape[0] < self.window+2*self.window_overlap:
			return

		# constant q transform with 96 half-tones from D1
		# in theory we only need notes from D2-D6, but in practice tuning is often too low
		# and harmonics are needed above D6
		cqt= np.abs(librosa.cqt(
			y=self.buffer,
			sr= self.sr,
			hop_length= self.hop_length,
			fmin= 36.71,
			n_bins = 96))
		onset_env_cqt= librosa.onset.onset_strength(
			sr=self.sr,
			S=librosa.amplitude_to_db(cqt, ref=np.max))
		onsets_cqt_raw= librosa.onset.onset_detect(
			y= self.buffer,
			sr= self.sr,
			hop_length= self.hop_length,
			onset_envelope= onset_env_cqt,
			units= 'time',
			backtrack= True,
			wait= 0.1*self.sr/self.hop_length,
			delta= 0.2)

		onsets_cqt= filter(lambda t: t > self.window_overlap_t and t < self.window_t + self.window_overlap_t, onsets_cqt_raw)

		p= PointStamped()
		p.point.y= 0.2
		p.header.stamp= self.buffer_time+rospy.Duration(self.window_overlap_t)
		self.pub.publish(p)
		for o in onsets_cqt:
			p= PointStamped()
			p.header.stamp= self.buffer_time+rospy.Duration(o)
			p.point.x= 0.5
			rospy.loginfo('found onset at time {}'.format(p.header.stamp))
			self.pub.publish(p)

		self.buffer_time+= rospy.Duration(self.window_t)
		self.buffer= self.buffer[(-2*self.window_overlap):]

		if rospy.Time.now()-now > rospy.Duration(self.window):
			rospy.logerr('computation took longer than processed window')

def main():
	rospy.init_node('detect_onset')

	detector = OnsetDetector()
	rospy.spin()

if __name__ == '__main__':
	main()
