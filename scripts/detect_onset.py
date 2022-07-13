#!/usr/bin/env python

import rospy
import cv_bridge

from audio_common_msgs.msg import AudioData, AudioInfo
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

import librosa

import cv2
import numpy as np

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

		self.cv_bridge= cv_bridge.CvBridge()
		self.spectrogram= None
		self.previous_onsets= []
		self.pub_spectrogram= rospy.Publisher('spectrogram', Image, queue_size= 1)

		self.pub_compute_time= rospy.Publisher('~compute_time', Float32, queue_size= 1)

		self.last_time = rospy.Time.now()

	def reset(self):
		# audio buffer
		self.buffer_time= None
		self.buffer= self.buffer[0:0]

		# visualization
		self.spectrogram= None
		self.previous_onsets= []

	def update_spectrogram(self, spec, onsets):
		if self.pub_spectrogram.get_num_connections() == 0:
			self.spectrogram= None
			return

		overlap_hops= int(self.window_overlap/self.hop_length)

		# throw away overlap
		spec= spec[:, overlap_hops:-overlap_hops]
		onsets= [o-self.window_overlap_t for o in onsets]

		log_spec= np.log(spec)

		if self.spectrogram is None:
			self.spectrogram= log_spec
			return
		elif self.spectrogram.shape[1] > spec.shape[1]:
			self.spectrogram= self.spectrogram[:, -spec.shape[1]:]
		self.spectrogram= np.concatenate([self.spectrogram, log_spec],1)

		spectrogram= np.array(self.spectrogram/np.max(self.spectrogram)*255, dtype= np.uint8)
		heatmap = cv2.applyColorMap(spectrogram, cv2.COLORMAP_JET)
		LINECOLOR=[255,0,255]
		for o in self.previous_onsets:
			heatmap[:,int(o*self.sr/self.hop_length)][:] = LINECOLOR
		for o in onsets:
			heatmap[:,int(self.window/self.hop_length + o*self.sr/self.hop_length)][:]= LINECOLOR
		self.previous_onsets= onsets

		self.pub_spectrogram.publish(self.cv_bridge.cv2_to_imgmsg(heatmap, "bgr8"))

	def audio_cb(self, msg):
		# handle bag loop graciously
		now = rospy.Time.now()
		if now < self.last_time:
			rospy.loginfo('detected bag loop')
			self.reset()
			self.last_time= now

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

		onsets_cqt= [o for o in onsets_cqt_raw if o >= self.window_overlap_t and o < self.window_t + self.window_overlap_t]

		self.update_spectrogram(cqt, onsets_cqt)

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
		if len(onsets_cqt) == 0:
			rospy.loginfo('found no onsets')

		self.buffer_time+= rospy.Duration(self.window_t)
		self.buffer= self.buffer[(-2*self.window_overlap):]

		compute_time= rospy.Time.now() - now
		self.pub_compute_time.publish(compute_time.to_sec())
		if compute_time > rospy.Duration(self.window):
			rospy.logerr('computation took longer than processed window')

def main():
	rospy.init_node('detect_onset')

	detector = OnsetDetector()
	rospy.spin()

if __name__ == '__main__':
	main()
