import copy
import numpy as np
import os
import pandas as pd
import rospy
import sklearn.gaussian_process as gp

from nav_msgs.msg import Path
from .paths import RuckigPath
from .utils import note_to_string, publish_figure

import matplotlib.pyplot as plt
plt.switch_backend('agg')


class OnsetToPath:
    def __init__(self, storage = '/tmp/plucks.json'):
        self.pluck_table = pd.DataFrame(
            columns=(*RuckigPath().params_map.keys(), 'finger', 'loudness', 'detected_note', 'onset_cnt', 'onsets')
            )
        self.storage = storage
        if os.path.exists(self.storage):
            self.pluck_table = pd.read_json(self.storage)

        self.print_summary()

        rospy.on_shutdown(self.store_plucks)

    def store_plucks(self):
        rospy.loginfo(f"storing plucks in '{self.storage}'")
        self.store_to_file()

    def print_summary(self):
        summary= f"OnsetToPath stores {len(self.pluck_table)} plucks\n"
        for n in set(self.pluck_table['detected_note']):
            summary+= f"{n}: {len(self.pluck_table[self.pluck_table['detected_note'] == n])} plucks\n"
        rospy.loginfo(summary)

    def store_to_file(self):
        self.pluck_table.to_json(self.storage)

    def add_sample(self, row):
        row_df = pd.DataFrame(row, columns= row.keys(), index= [0])
        self.pluck_table = pd.concat((self.pluck_table, row_df), ignore_index=True)

    def infer_next_best_pluck(self, *, string, finger, string_length, direction):
        plucks = self.pluck_table[
            (self.pluck_table['note'] == string) &
            (self.pluck_table['finger'] == finger) &
            (self.pluck_table['post_y']*direction >= 0.0)
        ]

        if len(plucks) < 3:
            return None

        features = plucks[['string_position', 'keypoint_pos_y']].to_numpy()
        # features_mean = features.mean(axis=0)
        features_mean = np.array([string_length/2, 0.0])
        # features_std = features.std(axis=0)
        features_std = np.array([string_length/4, 0.004])

        loudness = plucks['loudness'].to_numpy()

        features = (features - features_mean) / features_std

        GPR= gp.GaussianProcessRegressor(n_restarts_optimizer=10, alpha=0.5**2)

        GPR.fit(features, loudness)

        # TODO: sample randomly or CEM instead?
        grid_size = 100
        # coordinates
        pos_limits= np.array([-0.004, 0.008])
        if direction < 0.0:
            pos_limits= -1.0*pos_limits[::-1]

        xi, yi = np.meshgrid(
            np.linspace(0.0, string_length, grid_size),
            np.linspace(pos_limits[0], pos_limits[1], grid_size)
            )
        xi= ((xi- features_mean[0])/features_std[0]).ravel()
        yi= ((yi - features_mean[1])/features_std[1]).ravel()
        grid_points = np.column_stack((xi, yi))
        means, std = GPR.predict(grid_points, return_std=True)
        max_std= np.argmax(std)
        max_std= np.array([xi[max_std], yi[max_std]])
        max_std= max_std*features_std + features_mean

        fig = plt.figure(dpi= 50)
        means = means.reshape((grid_size, grid_size))
        plt.imshow(means, origin='lower', cmap=plt.get_cmap("RdPu"))
        plt.colorbar()
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        publish_figure("gp_loudness", fig)
        fig = plt.figure(dpi= 50)
        std = std.reshape((grid_size, grid_size))
        plt.imshow(std, origin='lower', cmap=plt.get_cmap("RdPu"))
        plt.colorbar()
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        publish_figure("gp_std_loudness", fig)
        return max_std

    def get_note_min_max(self, note : str):
        '''
        Returns the minimum and maximum loudness for the given note.
        '''
        note_plucks = self.pluck_table[
            (self.pluck_table['note'] == note_to_string(note)) &
            (self.pluck_table['detected_note'] == note) &
            (self.pluck_table['onset_cnt'] == 1)]
        if len(note_plucks) == 0:
            raise ValueError(f"No plucks found for note {note}")
        return np.min(note_plucks['loudness']), np.max(note_plucks['loudness'])

    def get_path(self, note : str, loudness : float, finger : str = None, direction = 0.0, string_position= None) -> Path:
        '''
        Returns a path for the given note and loudness.

        @param note: The note to play
        @param loudness: The loudness to play the note at
        @param direction: The direction to pluck in (1 for towards the robot, -1 for away from the robot)
        '''
        note_plucks = self.pluck_table[
            (self.pluck_table['onset_cnt'] == 1) &
            (self.pluck_table['note'] == note_to_string(note)) &
            (self.pluck_table['detected_note'] == note) &
            (self.pluck_table['post_y']*direction >= 0.0)
        ]

        if len(note_plucks) == 0:
            raise ValueError(f"No plucks found for note {note} in direction {direction}")

        if finger is not None:
            note_plucks = note_plucks[note_plucks['finger'] == finger]
            if len(note_plucks) == 0:
                raise ValueError(f"No plucks found for note {note} and finger {finger}")

        objective = np.abs(note_plucks['loudness'] - loudness)
        # if string_position is not None:
        #     objective+= 10*np.abs(note_plucks['string_position']-string_position)

        pluck = note_plucks.iloc[np.argmin(objective)]

        if string_position is not None:
            pluck= copy.deepcopy(pluck)
            pluck['string_position'] = string_position
        return RuckigPath.from_map(pluck)(), pluck['finger']
