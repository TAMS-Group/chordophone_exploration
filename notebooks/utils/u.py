import rospy
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

from IPython.display import HTML

try:
    mpl.style.use('seaborn-v0_8')
except:
    pass
#plt.rcParams['font.sans-serif'] = ['Arial'] # support MUSIC SHARP

import numpy as np
import struct

from collections import Counter

from functools import reduce

import librosa
from tams_pr2_guzheng.msg import PluckEpisodeV2
from audio_common_msgs.msg import AudioData

from IPython import display

# constants

guzheng_range = reduce(lambda a,b: a+b, list(map(lambda o: [f"{n}{o}" for n in ["d", "e", "fis", "a", "b"]], range(2, 6)))) + ["d6"]
cqt_range = reduce(lambda a,b: a+b, list(map(lambda o: [f"{n}{o}" for n in ["c", "cis", "d", "dis", "e", "f", "fis", "g", "gis", "a", "ais", "b"]], range(2, 8)))) + ["c8"]

magic_cqt_offset= 0.0

# utilities

def note_to_string(n):
    return n.lower().replace('♯', 'is')

def string_to_note(s):
    return s.upper().replace('IS', '♯')

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
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(f'plots/{s}.png', dpi= 300)

def save_anim(a, s):
    if not os.path.exists('plots'):
        os.makedirs('plots')
    a.save(f'plots/{s}.mp4', dpi= 150)
    return HTML(a.to_html5_video())

def big_plot():
    plt.figure(figsize=(15,8), dpi=100)

def label_outer():
    for ax in plt.gcf().axes:
        try:
            ax.label_outer()
        except:
            pass

def plot_cqt_energy(e):
    data_start = (e.cqt.header.stamp - e.start_execution).to_sec()
    cqt = cqt_from_episode(e).T
    plt.plot(
        data_start + np.arange(cqt.shape[1])*e.cqt.hop_length/e.cqt.sample_rate +magic_cqt_offset,
        cqt.sum(axis=0),
        'o-'
        )

def plot_aligned_audio_tactile(e, context= 0.3):
    plt.suptitle("Aligned Modalities")

    cqt = cqt_from_episode(e).T[target_cqt_idx(e)-2:target_cqt_idx(e)+3,:]

    plots = [
        ("raw audio", lambda: plot_raw_audio(e)),
        ("cqt", lambda: plot_cqt(e)),
        ("cqt\ntarget", lambda: plot_cqt(e, cqt)),
        ("cqt\nenergy", lambda: plot_cqt_energy(e)),
        ("target\nnote", lambda: plot_target_note(e)),
        ("pdc", lambda: plot_tactile(e)),
        ("pac", lambda: plot_tactile_ac(e)),
        (f"{e.finger.upper()}J3\npos", lambda: plot_joint_pos(e, f"rh_{e.finger.upper()}J3")),
        (f"{e.finger.upper()}J2\npos", lambda: plot_joint_pos(e, f"rh_{e.finger.upper()}J2")),
    ]

    N = len(plots)
    ax = None
    for n in range(0,N):
        ax = plt.subplot(N,1,n+1, sharex=ax)
        plots[n][1]()
        plt.ylabel(plots[n][0])

    plt.xlabel('time')
    label_outer()

    os = [p.header.stamp.to_sec() for p in e.detected_tactile_plucks]+[o.header.stamp.to_sec() for o in e.detected_audio_onsets]
    if len(os) == 0:
        os= [(e.header.stamp+e.length/2).to_sec()]
    plt.xlim((min(os)-context)-e.start_execution.to_sec(), (max(os)+context)-e.start_execution.to_sec())


def plot_episode(e, joints= True):
    plt.figure(figsize=(7,3), dpi=100)
    plt.title('audio recording')
    plot_raw_audio(e)

    plt.figure(figsize=(5, 5), dpi=100)
    plt.title('tip paths')
    plot_tip_path(e)

    plt.figure()
    plot_aligned_audio_tactile(e)

    if joints:
        #plt.figure(figsize=(7,15), dpi=100)
        plot_joints(e)

def plot_tip_path(e):
    plt.title(f"{e.finger} tip path near string {e.string}")
    plt.gca().set_aspect('equal', 'box')
    plt.plot([-p.pose.position.y for p in e.commanded_path.poses], [p.pose.position.z for p in e.commanded_path.poses], color='r')
    plt.plot([-p.pose.position.y for p in e.planned_path.poses], [p.pose.position.z for p in e.planned_path.poses], color='g')
    plt.plot([-p.pose.position.y for p in e.executed_path.poses], [p.pose.position.z for p in e.executed_path.poses], color='b')
    plt.legend(['commanded', 'planned', 'executed'])

def plot_joint_pos(e, joint):
    plt.plot(*joint_positions(e.executed_trajectory, joint), 'o-', color='b')
    # plt.ylabel(f"{joint}\nposition")

def plot_joint_vel(e, joint):
    plt.plot(*joint_velocities(e.executed_trajectory, joint), color='b')
    # plt.ylabel(f"{joint}\nvelocity")

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
        plot_joint_pos(e, j)

        plt.subplot(9, 2, idx)
        idx+= 1
        plot_joint_vel(e, j)

    plt.tight_layout()

def plot_cqt(e, cqt= None, onsets= True):
    data_start = (e.cqt.header.stamp - e.start_execution).to_sec()

    if cqt is None:
        cqt = cqt_from_episode(e).T

    X= np.tile((data_start + np.arange(cqt.shape[1])*e.cqt.hop_length/e.cqt.sample_rate +magic_cqt_offset)[:, np.newaxis], cqt.shape[0]).T
    Y= np.tile(np.arange(cqt.shape[0])[:,np.newaxis], cqt.shape[1])

    plt.pcolormesh(X, Y, cqt, cmap='jet')
    if onsets:
        plot_onsets(e)

def target_cqt_idx(e : PluckEpisodeV2) -> int:
    return int(librosa.note_to_midi(string_to_note(e.string)) - librosa.note_to_midi(e.cqt.min_note))

def plot_target_note(e):
    data_start = (e.cqt.header.stamp-e.start_execution).to_sec()
    cqt = cqt_from_episode(e).T
    plt.plot(data_start + np.arange(cqt.shape[1])*e.cqt.hop_length/e.cqt.sample_rate, cqt[target_cqt_idx(e),:], 'o-')
    plot_onsets(e)

def harmonics_for_index(fundamental_note_idx):
    return np.array([fundamental_note_idx + i for i in (0, 12, 19, 24, 28, 31, 35, 36) if fundamental_note_idx + i < len(cqt_range)])

def plot_harmonics(e, ax= None):
    if ax is None:
        ax= plt.gca()

    ax.set_prop_cycle(None)

    fundamental_note_idx = cqt_range.index(e.string)
    # fundamental_note_idx = u.cqt_range.index('b3')
    harmonics = harmonics_for_index(fundamental_note_idx)

    cqt= cqt_from_episode(e).T
    cqt_t= (e.cqt.header.stamp - e.start_execution).to_sec() + np.arange(cqt.shape[1])*e.cqt.hop_length/e.cqt.sample_rate
    cqt= cqt[harmonics, :]
    plot = ax.plot(
        cqt_t,
        cqt.T)
    plt.xlim([0, max(cqt_t)])
    plt.xlabel('time (hops)')
    plt.ylabel('cqt')
    plt.legend([cqt_range[h] for h in harmonics], loc='upper left')
    return plot

def plot_harmonics_bars(e, note= None, cqt= None, ax= None):
    if ax is None:
        ax= plt.gca()

    ax.set_prop_cycle(None)

    if note is None:
        note= e.string
    fundamental_note_idx = cqt_range.index(note)
    harmonics = harmonics_for_index(fundamental_note_idx)
    if cqt is None:
        cqt= cqt_from_episode(e)

    # reset colors
    ax.set_prop_cycle(None)
    for i in reversed(range(len(harmonics)-1)):
        plt.bar(
            range(cqt.shape[0]),
            cqt[:,harmonics[:i+1]].sum(axis=-1)
            )
    plt.xticks(
        range(cqt.shape[0]),
        [(f'{(i-2)*e.cqt.hop_length/e.cqt.sample_rate:.2F}' if (i+3)%5==0 else '') for i in range(0, cqt.shape[0])]
        )
    plt.xticks(rotation= 90)

    plt.xlabel('time (s)')
    plt.ylabel('cqt')
    plt.legend([cqt_range[h] for h in reversed(harmonics)], loc='upper right')

def audio_from_episode(e):
    return np.frombuffer(e.audio_data.audio.data, dtype=np.int16).astype(float)

def plot_raw_audio(e, onsets = True):
    data_start = (e.audio_data.header.stamp-e.start_execution).to_sec()

    signal = np.array(struct.unpack('{0}h'.format(int(len(e.audio_data.audio.data)/2)), e.audio_data.audio.data), dtype=float)
    plt.plot(data_start + np.arange(len(signal), dtype=float)/e.audio_info.sample_rate, signal)
    if onsets:
        plot_onsets(e)

def plot_tactile(e):
    plt.plot([ (t.header.stamp-e.start_execution).to_sec() for t in e.tactile_data],[t.tactile.pdc for t in e.tactile_data])
    plot_onsets(e)

def plot_tactile_ac(e):
    plt.plot([ (t.header.stamp-e.start_execution).to_sec() for t in e.tactile_data],[t.tactile.pac0 for t in e.tactile_data])
    plot_onsets(e)

def plot_joint(e, joint):
    j_idx = e.executed_trajectory.joint_names.index(joint)
    sample_times = [p.time_from_start.to_sec()-e.start_execution.to_sec() for p in e.executed_trajectory.points]+[(e.header.stamp+e.length).to_sec()]
    sample_pos = [p.positions[j_idx] for p in e.executed_trajectory.points]+[e.executed_trajectory.points[-1].positions[j_idx]]
    plt.plot(sample_times, sample_pos, color='b')
    plot_onsets(e)

def plot_onsets(e):
    for o in [o.header.stamp+rospy.Duration(magic_cqt_offset) for o in e.detected_audio_onsets]:
        plt.axvline((o-e.start_execution).to_sec(), ymin= 0.05, ymax= 0.95, color= 'purple')

    for p in [p.header.stamp for p in e.detected_tactile_plucks]:
        plt.axvline((p-e.start_execution).to_sec(), ymin= 0.05, ymax= 0.95, color= 'red')

def describe_episode(e):
    header= f'Episode {e.id}'
    print(f'{header}\n{"-"*len(header)}\n\n'
        f'start: {e.header.stamp.to_sec()}\n'
        f'length: {e.length.to_sec()}\n')

    e_avg_sample_dt= ((e.executed_trajectory.points[-1].time_from_start - e.executed_trajectory.points[0].time_from_start)/len(e.executed_trajectory.points)).to_sec()
    print(f'executed trajectory sampled at {1/e_avg_sample_dt}\n')

    print(f'onsets: {len(e.detected_audio_onsets)}')
    for o in e.detected_audio_onsets:
        print(f'       - {o.note} at {(o.header.stamp-e.start_execution).to_sec():.6} with {o.loudness:.2F}dB')
    print(f'plucks: {len(e.detected_tactile_plucks)}')
    for p in e.detected_tactile_plucks:
        print(f'       - {(p.header.stamp-e.start_execution).to_sec():.6} {p.finger} with strength {p.strength:.2F}')

def show_animation(anim):
    video= anim.to_html5_video()
    html = display.HTML(video)
    display.display(html)
    plt.close()

def interactive_animation(anim):
    html = HTML(anim.to_jshtml())
    display.display(html)
    plt.close()

# Audio - requires `roslaunch tams_pr2_guzheng play_audio_topic.launch`

def init_audio():
    global audio_pub
    import audio_common_msgs.msg
    audio_pub= rospy.Publisher('guzheng/audio', audio_common_msgs.msg.AudioData, queue_size=10)
init_audio()

def play_audio(e):
    if hasattr(e, 'audio_data'):
        audio_pub.publish(e.audio_data.audio)
    elif isinstance(e, bytes):
        A= AudioData(data= e)
        audio_pub.publish(A)
    else:
        raise RuntimeError(f"can't play {type(e)}")
