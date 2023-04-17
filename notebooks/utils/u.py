import rospy
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn-v0_8')

import numpy as np
import struct

from collections import Counter

from functools import reduce

# constants

guzheng_range = reduce(lambda a,b: a+b, list(map(lambda o: [f"{n}{o}" for n in ["d", "e", "fis", "a", "b"]], range(2, 6)))) + ["d6"]
cqt_range = reduce(lambda a,b: a+b, list(map(lambda o: [f"{n}{o}" for n in ["c", "cis", "d", "dis", "e", "f", "fis", "g", "gis", "a", "ais", "b"]], range(2, 8)))) + ["c8"]

magic_cqt_offset= 0.0

# utilities

def note_to_string(n):
    return n.lower().replace('♯', 'is')

def cqt_from_episode(e):
    return np.array(e.cqt.data).reshape((-1,e.cqt.number_of_semitones))

def joint_positions(traj, joint):
    return list(zip(*[(p.time_from_start.to_sec(), p.positions[traj.joint_names.index(joint)]) for p in traj.points]))

def joint_velocities(traj, joint):
    return list(zip(*[(p.time_from_start.to_sec(), p.velocities[traj.joint_names.index(joint)]) for p in traj.points]))

def tip_path(path):
    return list(zip(*[(-p.pose.position.y, p.pose.position.z) for p in path.poses]))

# plotting

def save_plot(s):
    plt.savefig(f'plots/{s}.png', dpi= 150)

def save_anim(a, s):
    a.save(f'plots/{s}.mp4', dpi= 150)
    
def big_plot():
    plt.figure(figsize=(15,8), dpi=100)

def label_outer():
    for ax in plt.gcf().axes:
        try:
            ax.label_outer()
        except:
            pass
    

def plot_episode(e, joints= True):
    plt.figure(figsize=(7,3), dpi=100)
    plt.title('audio recording')
    plot_raw_audio(e)

    plt.figure(figsize=(5, 5), dpi=100)
    plt.title('tip paths')
    plot_tip_path(e)

    plt.figure()
    plt.suptitle("audio/tactile onsets")
    N=4
    ax = plt.subplot(N,1,1)
    plot_raw_audio(e)
    plt.ylabel("raw audio")
    context= 0.2
    os = [o.to_sec() for o in e.detected_tactile_plucks]+[o.header.stamp.to_sec() for o in e.detected_audio_onsets]
    if len(os) == 0:
        os= [(e.header.stamp+e.length/2).to_sec()]
    plt.xlim((min(os)-context)-e.start_execution.to_sec(), (max(os)+context)-e.start_execution.to_sec())

    ax = plt.subplot(N,1,2, sharex=ax)
    plot_audio(e)
    plt.ylabel("cqt")

    ax = plt.subplot(N,1,3, sharex=ax)
    plot_tactile(e)
    plt.ylabel("pdc")

    ax = plt.subplot(N,1,4, sharex=ax)
    plot_tactile_ac(e)
    plt.ylabel("pac")

    plt.xlabel('time')
    label_outer()

    if joints:
        plt.figure(figsize=(7,15), dpi=100)
        plot_joints(e)

def plot_tip_path(e):
    plt.title(f"{e.finger} tip path near string {e.string}")
    plt.gca().set_aspect('equal', 'box')
    plt.plot([-p.pose.position.y for p in e.commanded_path.poses], [p.pose.position.z for p in e.commanded_path.poses], color='r')
    plt.plot([-p.pose.position.y for p in e.planned_path.poses], [p.pose.position.z for p in e.planned_path.poses], color='g')
    plt.plot([-p.pose.position.y for p in e.executed_path.poses], [p.pose.position.z for p in e.executed_path.poses], color='b')
    plt.legend(['commanded', 'planned', 'executed'])

def plot_joints(e):
    right_arm_executed_joints = ['r_shoulder_pan_joint','r_shoulder_lift_joint','r_upper_arm_roll_joint','r_elbow_flex_joint','r_forearm_roll_joint','rh_WRJ2','rh_WRJ1']

    plt.figure(figsize=(10,15), dpi=75)
    plt.suptitle('velocities of planned vs executed trajectory')
    for i, j in enumerate(right_arm_executed_joints):
        plt.subplot(9, 2, 2*i+1)
        plt.title(f"{j} positions")
        plt.plot(*joint_positions(e.planned_trajectory, j), color='r')
        plt.plot(*joint_positions(e.executed_trajectory, j), color='b')
    
        plt.subplot(9, 2, 2*(i+1))
        plt.title(f"{j} velocities")
        plt.plot(*joint_velocities(e.planned_trajectory, j), color='r')
        plt.plot(*joint_velocities(e.executed_trajectory, j), color='b')

    idx= 15
    for j in ["rh_FFJ3", "rh_FFJ2"]:
        plt.subplot(9, 2, idx)
        idx+= 1
        plt.title(f"{j} positions")
        plt.plot(*joint_positions(e.executed_trajectory, j), color='b')

        plt.subplot(9, 2, idx)
        idx+= 1
        plt.title(f"{j} velocity")
        plt.plot(*joint_velocities(e.executed_trajectory, j), color='b')

    plt.tight_layout()

    
def plot_audio(e):
    cqt = np.log(cqt_from_episode(e)).T
    #plt.imshow(cqt, cmap='jet')
    plt.grid(False)
    
    X= np.tile(np.arange(84).T[:,np.newaxis], cqt.shape[1])
    Y= np.tile((np.arange(cqt.shape[1])*512/44100 + (e.cqt.header.stamp - e.start_execution).to_sec()+magic_cqt_offset)[:, np.newaxis], cqt.shape[0]).T
    
    plt.pcolormesh(Y, X, cqt, cmap='jet')
    
    plt.vlines([(o.header.stamp-e.start_execution).to_sec()+magic_cqt_offset for o in e.detected_audio_onsets], ymin=0, ymax= 84, color=(1.0,0,1.0,0.8))
    plt.vlines([(o-e.start_execution).to_sec() for o in e.detected_tactile_plucks], ymin=0-4, ymax= 84+4, color='red')

def plot_raw_audio(e):
    signal = np.array(struct.unpack('{0}h'.format(int(len(e.audio_data.audio.data)/2)), e.audio_data.audio.data), dtype=float)
    
    plt.plot(np.arange(len(signal), dtype=float)/e.audio_info.sample_rate+e.audio_data.header.stamp.to_sec() - e.start_execution.to_sec(), signal)

    vmin = np.min(signal)*1.05
    vmax = np.max(signal)*1.05

    plt.vlines([(o.header.stamp - e.start_execution).to_sec()+magic_cqt_offset for o in e.detected_audio_onsets], ymin=vmin, ymax=vmax, color='purple')
    plt.vlines([o.to_sec() - e.start_execution.to_sec() for o in e.detected_tactile_plucks], ymin=vmin, ymax= vmax, color='red')

def plot_tactile(e):
    plt.plot([t.header.stamp.to_sec()-e.start_execution.to_sec() for t in e.tactile_data],[t.tactile.pdc for t in e.tactile_data])
    vmin = np.min([t.tactile.pdc for t in e.tactile_data])
    vmax = np.max([t.tactile.pdc for t in e.tactile_data])
    delta = (vmax - vmin)*0.05
    vmin = vmin - delta
    vmax = vmax + delta
    plt.vlines([o.header.stamp.to_sec()-e.start_execution.to_sec()+magic_cqt_offset for o in e.detected_audio_onsets],
               vmin,
               vmax,
               'purple')
    plt.vlines([o.to_sec()-e.start_execution.to_sec() for o in e.detected_tactile_plucks],
               vmin,
               vmax, 'red')

def plot_tactile_ac(e):
    plt.plot([t.header.stamp.to_sec()-e.start_execution.to_sec() for t in e.tactile_data],[t.tactile.pac0 for t in e.tactile_data])
    vmin = np.min([t.tactile.pac0 for t in e.tactile_data])
    vmax = np.max([t.tactile.pac0 for t in e.tactile_data])
    delta = (vmax - vmin)*0.05
    vmin = vmin - delta
    vmax = vmax + delta
    plt.vlines([o.header.stamp.to_sec()-e.start_execution.to_sec()+magic_cqt_offset for o in e.detected_audio_onsets],
               vmin,
               vmax,
               'purple')
    plt.vlines([o.to_sec()-e.start_execution.to_sec() for o in e.detected_tactile_plucks],
               vmin,
               vmax, 'red')

    
def plot_joint(e, joint):
    j_idx = e.executed_trajectory.joint_names.index(joint)
    sample_times = [p.time_from_start.to_sec()-e.start_execution.to_sec() for p in e.executed_trajectory.points]+[(e.header.stamp+e.length).to_sec()]
    sample_pos = [p.positions[j_idx] for p in e.executed_trajectory.points]+[e.executed_trajectory.points[-1].positions[j_idx]]
    plt.plot(sample_times, sample_pos, color='b')
    
    vmin = np.min(sample_pos)
    vmax = np.max(sample_pos)
    delta = (vmax - vmin)*0.5
    vmin = vmin - delta
    vmax = vmax + delta
    plt.vlines([o.header.stamp.to_sec()-e.start_execution.to_sec()+magic_cqt_offset for o in e.detected_audio_onsets],
               vmin,
               vmax,
               'purple')
    plt.vlines([o.to_sec()-e.start_execution.to_sec() for o in e.detected_tactile_plucks],
               vmin,
               vmax, 'red')

# Audio - requires `roslaunch tams_pr2_guzheng play_audio_topic.launch`

def init_audio():
    global audio_pub
    import audio_common_msgs.msg
    audio_pub= rospy.Publisher('guzheng/audio', audio_common_msgs.msg.AudioData, queue_size=10)
init_audio()

def play_audio(e):
    audio_pub.publish(e.audio_data.audio)