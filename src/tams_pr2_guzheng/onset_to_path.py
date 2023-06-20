import copy
import numpy as np
import os
import pandas as pd
import rospy
import sklearn.gaussian_process as gp

from typing import NamedTuple, Tuple

from nav_msgs.msg import Path
from .paths import RuckigPath
from .utils import note_to_string, publish_figure

import matplotlib.pyplot as plt
plt.switch_backend('agg')

class OnsetToPath:
    class ActionSpace(NamedTuple):
        string_position: np.array
        keypoint_pos_y: np.array

        def is_valid(self, plucks):
            # TODO: string_position varies!
            return np.logical_and.reduce((
                plucks['string_position'] >= self.string_position[0],
                plucks['string_position'] <= self.string_position[1],
                plucks['keypoint_pos_y'] >= self.keypoint_pos_y[0],
                plucks['keypoint_pos_y'] <= self.keypoint_pos_y[1],
            ))

    def __init__(self, *, storage : str = '/tmp/plucks.json'):
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
            if np.isnan(n):
                continue
            summary+= f"{n}: {len(self.pluck_table[self.pluck_table['detected_note'] == n])} plucks\n"
        summary+= f"nan: {len(self.pluck_table[np.isnan(self.pluck_table['detected_note'])])} plucks\n\n"

        for n in set(self.pluck_table['string']):
            summary+= f"{n}: {len(self.pluck_table[self.pluck_table['string'] == n])} plucks\n"

        rospy.loginfo(summary)

    def store_to_file(self):
        self.pluck_table.to_json(self.storage)

    def add_sample(self, row):
        row_df = pd.DataFrame(row, columns= row.keys(), index= [0])
        self.pluck_table = pd.concat((self.pluck_table, row_df), axis= 0, ignore_index=True)

    class ActionParameters(NamedTuple):
        string_position : float
        keypoint_pos_y : float

    def infer_next_best_pluck(self, *, string : str, finger : str, actionspace : ActionSpace, direction : float):
        assert(direction in [-1.0, 1.0])

        plucks = self.pluck_table[
            (self.pluck_table['string'] == string) &
            (self.pluck_table['finger'] == finger) &
            (self.pluck_table['post_y']*direction >= 0.0)
        ]

        plucks = plucks[actionspace.is_valid(plucks)]

        if len(plucks) < 1:
            return None

        features = plucks[['string_position', 'keypoint_pos_y']].to_numpy()
        # features_mean = features.mean(axis=0)
        # features_std = features.std(axis=0)
        # use expected means/std instead:
        features_mean = np.array([actionspace.string_position[1]/2, 0.0])
        features_std = np.array([actionspace.string_position[1]/4, 0.004])

        loudness = plucks['loudness'].to_numpy().astype(np.float64)
        loudness[np.isnan(loudness)] = 0.0

        features = (features - features_mean) / features_std

        GPR= gp.GaussianProcessRegressor(n_restarts_optimizer=10, alpha=0.5**2)
        GPR.fit(features, loudness)

        # TODO: sample randomly or CEM instead?
        grid_size = 100

        # limits are always given as lower(closer to pre), higher(closer to post), so invert if needed
        pos_limits= actionspace.keypoint_pos_y
        if direction < 0.0:
            pos_limits= -1.0*pos_limits[::-1]

        xi, yi = np.meshgrid(
            np.linspace(actionspace.string_position[0], actionspace.string_position[1], grid_size),
            np.linspace(pos_limits[0], pos_limits[1], grid_size)
            )
        xi= ((xi- features_mean[0])/features_std[0]).ravel()
        yi= ((yi - features_mean[1])/features_std[1]).ravel()
        grid_points = np.column_stack((xi, yi))
        means, std = GPR.predict(grid_points, return_std=True)
        idx_max_std= np.argmax(std)
        features_norm_max_std= np.array([xi[idx_max_std], yi[idx_max_std]])
        features_max_std= features_norm_max_std*features_std + features_mean

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
        return self.ActionParameters(*features_max_std)

    def get_note_min_max(self, note : str):
        '''
        Returns the minimum and maximum loudness for the given note.
        '''
        note_plucks = self.pluck_table[
            (self.pluck_table['string'] == note_to_string(note)) &
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
            (self.pluck_table['string'] == note_to_string(note)) &
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
