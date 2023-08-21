import copy
import librosa
import numpy as np
import os
import pandas as pd
import rospy
import sklearn.gaussian_process as gp
import scipy.stats as stats
import tams_pr2_guzheng.utils as utils

from typing import NamedTuple, Tuple

from nav_msgs.msg import Path
from tams_pr2_guzheng.msg import RunEpisodeResult

from .paths import RuckigPath
from .utils import note_to_string, string_to_note, publish_figure

import matplotlib.pyplot as plt
import seaborn as sns

def normalize(x, params = None):
    if params is None:
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std == 0] = 1.0
        params = (mean, std)
        return (x - mean) / std, params
    else:
        (mean, std) = params
        return (x - mean) / std

def undo_normalize(x, params):
    (mean, std) = params
    return x * std + mean


class OnsetToPath:
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
        for n in self.pluck_table['detected_note'].unique():
            if np.isnan(n):
                continue
            summary+= f"{n}: {len(self.pluck_table[self.pluck_table['detected_note'] == n])} plucks\n"

        summary+= f"nan: {len(self.pluck_table[self.pluck_table['detected_note'].isnull()])} plucks\n\n"


        for n in set(self.pluck_table['string']):
            summary+= f"{n}: {len(self.pluck_table[self.pluck_table['string'] == n])} plucks\n"

        rospy.loginfo(summary)

    def store_to_file(self):
        self.pluck_table.to_json(self.storage)

    def add_sample(self, row):
        row['onsets'] = str(row['onsets']) # convert to string for eventual json serialization
        row_df = pd.DataFrame(row, columns= row.keys(), index= [0])
        self.pluck_table = pd.concat((self.pluck_table, row_df), axis= 0, ignore_index=True)

        self.plot_loudness_strips()
        self.plot_scores(row)

    def score_row(self, row):
        r = copy.deepcopy(row)
        try:
            del r['onsets'] # this field is a list of onsets, but pandas just ignores the whole dict if it sees it
        except KeyError:
            pass
        df = pd.DataFrame(r, columns= r.keys(), index= [0])

        return self.score(df)[0]

    def score(self, df):
        # minimum distance to neighbors to consider as safe
        safe_threshold = 0.006 # m
        # safe_threshold = 0.004 # m
        # distance to saturation of distance safety score
        saturation_threshold  = 0.01 # m
        # loudness cut-off
        loudness_threshold = 60.0 # dBA

        a = 1/(saturation_threshold-safe_threshold)
        b = -a*safe_threshold

        scores = (a*df['min_distance']+b).to_numpy()
        scores[df['min_distance'] >= saturation_threshold] = 1.0
        scores[df['loudness'] > loudness_threshold] = -0.5
        scores[df['unexpected_onsets'] > 0] = -1.0

        return scores

    def plot_scores(self, row):
        '''plot and publish safety scores related to pluck family of row'''

        # find related plucks
        plucks = self.pluck_table[
            (self.pluck_table['string'] == row['string']) &
            (self.pluck_table['finger'] == row['finger']) &
            (self.pluck_table['post_y']*row['post_y'] >= 0.0)
        ]

        if len(plucks) < 1:
            return

        # compute scores
        scores = self.score(plucks)
        backend= plt.get_backend()
        plt.switch_backend('agg')
        fig, ax = plt.subplots(dpi= 100)
        cmap = sns.color_palette("seismic", as_cmap=True)
        norm = plt.Normalize(vmin=-1.0, vmax=1.0)
        sns.scatterplot(x= plucks['string_position'], y= plucks['keypoint_pos_y'], hue= scores, palette= cmap, hue_norm= norm, legend= False, s= 100, ax= ax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=ax)
        publish_figure("scores", fig)
        plt.switch_backend(backend)

    def plot_loudness_strips(self):
        '''plot cross-string loudness overview'''

        # we want to keep the original order in pluck_table to notice temporal effects
        # so we prepare a copy for plotting
        X = self.pluck_table.copy()
        X= X.sort_values('string', key= lambda x: x.map(lambda a: librosa.note_to_midi(string_to_note(a))))
        X['direction'] = self.pluck_table['pre_y'].map(lambda y: 'inwards' if y < 0.0 else 'outwards')
        X['loudness'] = X['loudness'].fillna(-1.0)

        backend= plt.get_backend()
        plt.switch_backend('agg')
        fig = plt.figure(dpi= 100)
        ax : plt.Axes = sns.stripplot(x=X['string'], y=X['loudness'], hue= X['direction'], hue_order= ['inwards', 'outwards'], ax = fig.gca())
        ax.set_ylabel('loudness [dBA]')
        publish_figure("loudness_strips", fig)
        plt.switch_backend(backend)

    def fit_GPR(self, features, value):
        GPR= gp.GaussianProcessRegressor(n_restarts_optimizer=10, alpha=0.5**2)
        GPR.fit(features, value)
        return GPR

    def infer_next_best_pluck(self, *, string : str, finger : str, actionspace : RuckigPath.ActionSpace, direction : float) -> RuckigPath:
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
        # features, features_norm_params = normalize(features)
        # use expected means/std instead:
        features_norm_params = (
            np.array([actionspace.string_position[1]/2, (actionspace.keypoint_pos_y[0] + (actionspace.keypoint_pos_y[1]-actionspace.keypoint_pos_y[0])/2) ]),
            np.array([actionspace.string_position[1]/4, (actionspace.keypoint_pos_y[1] - actionspace.keypoint_pos_y[0])/4])
        )
        features = normalize(features, features_norm_params)

        loudness = plucks['loudness'].to_numpy().astype(np.float64)
        loudness[np.isnan(loudness)] = 0.0

        GPR= self.fit_GPR(features, loudness)

        # limits are always given as lower(closer to pre), higher(closer to post), so invert if needed
        pos_limits= actionspace.keypoint_pos_y
        if direction < 0.0:
            pos_limits= -1.0*pos_limits[::-1]

        # TODO: CEM? Second-Order Optimization?
        sample_size= 100

        xi = np.random.uniform(actionspace.string_position[0], actionspace.string_position[1], sample_size)
        yi = np.random.uniform(pos_limits[0], pos_limits[1], sample_size)
        point_set = np.column_stack((xi, yi))
        point_set = normalize(point_set, features_norm_params)
        means, std = GPR.predict(point_set, return_std=True)
        idx_max_std= np.argmax(std)
        features_max_std= np.array([xi[idx_max_std], yi[idx_max_std]])

        # optional visualization
        grid_size = 50
        grid_string_position = np.linspace(actionspace.string_position[0], actionspace.string_position[1], grid_size)
        grid_keypoint_pos_y = np.linspace(pos_limits[0], pos_limits[1], grid_size)

        xi, yi = np.meshgrid(
            grid_string_position,
            grid_keypoint_pos_y
            )
        xi= xi.ravel()
        yi= yi.ravel()
        grid_points = np.column_stack((xi, yi))
        grid_points = normalize(grid_points, features_norm_params)
        means, std = GPR.predict(grid_points, return_std=True)
        fig = plt.figure(dpi= 100)
        means = means.reshape((grid_size, grid_size))
        plt.imshow(means, origin='lower', cmap=plt.get_cmap("RdPu"))
        plt.colorbar()
        plt.grid(False)
        plt.xticks(range(len(grid_string_position)), labels= [(f"{p:.2F}" if i%10==0 else "") for (i,p) in enumerate(grid_string_position)])
        plt.yticks(range(len(grid_keypoint_pos_y)), labels= [(f"{p:.4F}" if i%10==0 else "") for (i,p) in enumerate(grid_keypoint_pos_y)])
        plt.xlabel("string position")
        plt.ylabel("keypoint pos y")
        publish_figure("gp_loudness", fig)
        fig = plt.figure(dpi= 50)
        std = std.reshape((grid_size, grid_size))
        plt.imshow(std, origin='lower', cmap=plt.get_cmap("RdPu"))
        plt.colorbar()
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        publish_figure("gp_std_loudness", fig)

        p = RuckigPath.prototype(string= string, direction= direction)
        p.string_position= features_max_std[0]
        p.keypoint_pos[0]= features_max_std[1]
        return p

    def get_note_min_max(self, note : str):
        '''
        Returns the minimum and maximum loudness for the given note.
        '''
        plucks = self.pluck_table[
            (self.pluck_table['string'] == note_to_string(note)) &
            (self.pluck_table['detected_note'] == note) &
            (self.pluck_table['onset_cnt'] == 1)]
        if len(plucks) == 0:
            raise ValueError(f"No plucks found for note {note}")

        audible_plucks = plucks[plucks['loudness'] > 0.0]
        if len(audible_plucks) == 0:
            return 0.0, 0.0
        else:
            return np.min(audible_plucks['loudness']), np.max(audible_plucks['loudness'])

    def get_sample(self, note : str, loudness : float, finger : str = None, direction = 0.0, string_position= None) -> Path:
        '''
        Returns a path for the given note and loudness.

        @param note: The note to play
        @param loudness: The loudness to play the note at
        @param direction: The direction to pluck in (1 for towards the robot, -1 for away from the robot)
        '''
        plucks = self.pluck_table[
            (self.pluck_table['string'] == note_to_string(note)) &
            (self.pluck_table['post_y']*direction >= 0.0)
        ]

        if len(plucks) == 0:
            raise ValueError(f"No plucks found for note {note} in direction {direction}")

        if finger is not None:
            plucks = plucks[plucks['finger'] == finger]
            if len(plucks) == 0:
                raise ValueError(f"No plucks found for note {note} and finger {finger}")

        objective = np.abs(plucks['loudness'] - loudness)
        # if string_position is not None:
        #     objective+= 10*np.abs(note_plucks['string_position']-string_position)

        pluck = plucks.iloc[np.argmin(objective)]

        if string_position is not None:
            pluck= copy.deepcopy(pluck)
            pluck['string_position'] = string_position
        return RuckigPath.from_map(pluck)(), pluck['finger']

    def get_path(self, note : str, finger : str, direction : float, loudness : float= None, string_position : float= None) -> Tuple[Path, str, float]:
        '''
        Returns a path for the given note and loudness.

        @param note: The note to play
        @param finger: The finger to play the note with
        @param direction: The direction to pluck in (1 for towards the robot, -1 for away from the robot)
        @param target_loudness: The target loudness to play the note at
        @param string_position: The current string position to stay close to
        '''

        assert(direction in [-1.0, 1.0])

        string = note_to_string(note)

        if finger is None:
            fingers = self.pluck_table[self.pluck_table['string'] == string]['finger'].unique()
            if len(fingers) == 0:
                raise ValueError(f"No plucks found for string {string}")

            finger = fingers[0]
            if len(fingers) > 1:
                rospy.loginfo(f"{string} could be plucked with fingers {', '.join(fingers)}. arbitrarily choosing {finger}")

        plucks = self.pluck_table[
            (self.pluck_table['string'] == string) # &
        ]
        if len(plucks) == 0:
            raise ValueError(f"No plucks found for note {note}")

        plucks = plucks[plucks['post_y']*direction >= 0.0]
        if len(plucks) == 0:
            raise ValueError(f"No plucks found for note {note} in direction {direction}")

        plucks = plucks[plucks['finger'] == finger]
        if len(plucks) == 0:
            raise ValueError(f"No plucks found for note {note} and finger {finger} in direction {direction}")

        if plucks[
            (plucks['detected_note'] == note) &
            (plucks['onset_cnt'] == 1)
        ].empty:
            rospy.logwarn(f"no recorded pluck for {note} in direction {direction} with finger {finger} actually triggered the note {note}. Computed pluck is ungrounded.")

        # prepare features / loudness
        features = np.column_stack((
            plucks['string_position'],
            plucks['keypoint_pos_y']
        ))
        features, features_norm_params = normalize(features)
        loudnesses = plucks['loudness'].to_numpy()
        loudnesses[np.isnan(loudnesses)] = 0.0

        GPR = self.fit_GPR(features, loudnesses)

        # predict grid of points between min and max features
        grid_size = 100
        string_limits = (np.min(plucks['string_position']), np.max(plucks['string_position']))
        keypoint_pos_y_limits = (np.min(plucks['keypoint_pos_y']), np.max(plucks['keypoint_pos_y']))

        if string_position is not None:
            context = 0.07 # m around string_position
            string_positions = np.linspace(np.max((0.0, string_position-context)), np.min((string_position+context, string_limits[1])), 10)
        else:
            string_positions = np.linspace(*string_limits, grid_size)

        keypoint_pos_ys = np.linspace(*keypoint_pos_y_limits, grid_size)
        xi, yi = np.meshgrid(string_positions, keypoint_pos_ys)
        xi= xi.ravel()
        yi= yi.ravel()
        grid_features = np.column_stack((xi, yi))
        grid_features_normalized = normalize(grid_features, features_norm_params)

        # pick point with highest pdf for target loudness
        means, stds = GPR.predict(grid_features_normalized, return_std=True)
        probs = stats.norm(means, stds).pdf(loudness)
        # TODO: trade-off pdf and distance to target
        features_idx = np.argmax(probs)
        #features_idx = np.argmin(np.abs(means-loudness))

        string_position= xi[features_idx]
        keypoint_pos_y= yi[features_idx]

        p = RuckigPath.prototype(string= string, direction= direction, string_position= string_position)
        p.keypoint_pos[0] = keypoint_pos_y

        return p, finger, probs[features_idx]
