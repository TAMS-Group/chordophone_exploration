#!/usr/bin/env python

import rospy
import cv_bridge

from audio_common_msgs.msg import AudioData, AudioInfo
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Header, ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker
from tams_pr2_guzheng.msg import NoteOnset, CQTStamped


import librosa
import crepe
import crepe.core

import cv2
import numpy as np
import matplotlib.pyplot as plt

import struct
import sys
import time


class OnsetDetector():
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

		self.fmin_note= 'C2'
		self.fmin= librosa.note_to_hz(self.fmin_note)
		self.semitones= 84
		self.fmax_note= 'C8'
		self.fmax= librosa.note_to_hz(self.fmax_note)

		self.cmap= plt.get_cmap('gist_rainbow').copy()
		self.cmap.set_bad((0,0,0,1)) # make sure they are visible

		# number of samples for analysis window and overlap regions between consecutive windows
		self.window_t= 1.0
		self.window_overlap_t= 0.5

		self.window= int(self.sr*self.window_t)
		self.window_overlap= int(self.sr*self.window_overlap_t)

		self.overlap_hops= int(self.window_overlap/self.hop_length)

		# preload model to not block the callback on first message
		# capacities: 'tiny', 'small', 'medium', 'large', 'full'
		self.crepe_model= 'full'
		crepe.core.build_and_load_model(self.crepe_model)

		if not self.check_audio_format():
			rospy.signal_shutdown('incompatible audio format')
			return

		self.last_time = rospy.Time.now()

		self.cv_bridge= cv_bridge.CvBridge()

		self.reset()

		self.pub_spectrogram= rospy.Publisher('spectrogram', Image, queue_size= 1, tcp_nodelay= True)

		self.pub_compute_time= rospy.Publisher('~compute_time', Float32, queue_size= 1, tcp_nodelay= True)

		self.pub_cqt= rospy.Publisher('cqt', CQTStamped, queue_size= 100, tcp_nodelay= True)
		self.pub_onset= rospy.Publisher('onsets', NoteOnset, queue_size= 100, tcp_nodelay= True)
		self.pub= rospy.Publisher('onsets_markers', MarkerArray, queue_size= 100, tcp_nodelay= True)

		self.sub= rospy.Subscriber('audio', AudioData, self.audio_cb, queue_size= 500, tcp_nodelay= True)

	def reset(self):
		# audio buffer
		self.buffer_time= None
		self.buffer= np.array([], dtype= float)

		# visualization
		self.spectrogram= None
		self.previous_onsets= []

	def update_spectrogram(self, spec, onsets):
		if self.pub_spectrogram.get_num_connections() == 0:
			self.spectrogram= None
			return

		# throw away overlap
		spec= spec[:, self.overlap_hops:-self.overlap_hops]
		onsets= [o-self.window_overlap_t for o in onsets]

		log_spec= np.log(spec)

		if self.spectrogram is None:
			self.spectrogram= log_spec
			return
		elif self.spectrogram.shape[1] > spec.shape[1]:
			self.spectrogram= self.spectrogram[:, -spec.shape[1]:]
		self.spectrogram= np.concatenate([self.spectrogram, log_spec],1)

		# normalizes per compute
		spectrogram= np.array(self.spectrogram/np.max(self.spectrogram)*255, dtype= np.uint8)
		## absolute normalization for log image
		#absolute_log_threshold= 12.0
		#spectrogram= np.array(np.minimum(255, self.spectrogram/absolute_log_threshold*255), dtype= np.uint8)

		heatmap = cv2.applyColorMap(spectrogram, cv2.COLORMAP_JET)
		LINECOLOR=[255,0,255]
		for o in self.previous_onsets:
			heatmap[:,int(o*self.sr/self.hop_length)][:] = LINECOLOR
		for o in onsets:
			heatmap[:,int(self.window/self.hop_length + o*self.sr/self.hop_length)][:]= LINECOLOR
		self.previous_onsets= onsets

		self.pub_spectrogram.publish(self.cv_bridge.cv2_to_imgmsg(heatmap, "bgr8"))

	def fundamental_frequency_for_onset(self, onset):
		prediction_averaging_window= 0.1*self.sr # at most self.window_overlap to make sure the data exists
		excerpt = self.buffer[int(onset*self.sr):int(onset*self.sr+prediction_averaging_window)]
		time, freq, confidence, _ = crepe.predict(excerpt, self.sr, viterbi= True, model_capacity= self.crepe_model, verbose= 0)

		confidence_threshold= 0.4
		confidence_mask = confidence > confidence_threshold

		thresholded_freq= freq[confidence_mask]
		if len(thresholded_freq) > 0:
			pitch= np.average(thresholded_freq, weights= confidence[confidence_mask])
			rospy.loginfo('found frequency {} ({})'.format(pitch, librosa.hz_to_note(pitch)))
			return pitch, np.max(confidence[confidence_mask])
		else:
			return 0.0, 0.0

	def color_from_freq(self, freq):
		if freq > 0.0:
			return ColorRGBA(*self.cmap((freq-self.fmin)/(self.fmax-self.fmin)))
		else:
			return ColorRGBA(*self.cmap.get_bad())

	def publish_cqt(self, cqt):
		msg= CQTStamped()
		msg.number_of_semitones= self.semitones
		msg.min_note= self.fmin_note
		msg.hop_length= rospy.Duration(self.hop_length/self.sr)

		msg.header.stamp= self.buffer_time+self.window_overlap_t
		msg.data= self.buffer[:, self.overlap_hops:-self.overlap_hops].flatten()
		self.pub_cqt.publish(msg)

	def audio_cb(self, msg):
		# handle bag loop graciously
		now = rospy.Time.now()
		if now < self.last_time:
			rospy.loginfo('detected bag loop')
			self.reset()
			self.last_time= now

		# initially set time from now, later increment with each audio msg
		if self.buffer_time is None:
			self.buffer_time= now

		self.buffer= np.concatenate([self.buffer, OnsetDetector.unpack_data(msg.data)])

		# aggregate buffer until window+overlaps are full
		if self.buffer.shape[0] < self.window+2*self.window_overlap:
			return

		# TODO: it would be better to do the computation below asynchronously to be sure not to loose audio data

		# constant q transform with 96 half-tones from C2
		# in theory we only need notes from D2-D6, but in practice tuning is often too low
		# and harmonics are needed above D6
		cqt= np.abs(librosa.cqt(
			y=self.buffer,
			sr= self.sr,
			hop_length= self.hop_length,
			fmin= self.fmin,
			n_bins = self.semitones))

		self.publish_cqt(cqt)

		onset_env_cqt= librosa.onset.onset_strength(
			sr=self.sr,
			S=librosa.amplitude_to_db(cqt, ref=np.max))
		onsets_cqt_raw= librosa.onset.onset_detect(
			y= self.buffer,
			sr= self.sr,
			hop_length= self.hop_length,
			onset_envelope= onset_env_cqt,
			units= 'time',
			backtrack= False,
			#wait= 0.1*self.sr/self.hop_length,
			delta= 4.0, normalize= False,
			)

		onsets_cqt= [o for o in onsets_cqt_raw if o >= self.window_overlap_t and o < self.window_t + self.window_overlap_t]

		self.update_spectrogram(cqt, onsets_cqt)

		markers= MarkerArray()

		# publish events and plot visualization
		for o in onsets_cqt:
			fundamental_frequency, confidence= self.fundamental_frequency_for_onset(o)
			t= self.buffer_time+rospy.Duration(o)

			no= NoteOnset()
			no.header.stamp= t
			if fundamental_frequency != 0.0:
				no.note = librosa.hz_to_note(fundamental_frequency)
				no.confidence = confidence
			self.pub_onset.publish(no)

			m= Marker()
			m.ns= "audio_onset"
			m.type= Marker.SPHERE
			m.action= Marker.ADD
			m.header.stamp= t
			m.scale.x= 0.005
			m.scale.y= 0.005
			m.scale.z= 0.005
			m.color= self.color_from_freq(fundamental_frequency)
			markers.markers.append(m)
		self.pub.publish(markers)

		if len(onsets_cqt) == 0:
			rospy.logdebug('found no onsets')
		else:
			rospy.logdebug('found {} onsets'.format(len(onsets_cqt)))

		# advance buffer, keep one overlap for next processing
		self.buffer_time+= rospy.Duration(self.window_t)
		self.buffer= self.buffer[(-2*self.window_overlap):]

		rospy.loginfo_once('onset detection is online')
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
