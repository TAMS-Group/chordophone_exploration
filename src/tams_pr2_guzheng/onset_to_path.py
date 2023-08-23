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
        safe_threshold = 0.0 # m
        # safe_threshold = 0.004 # m
        # distance to saturation of distance safety score
        saturation_threshold  = 0.015 # m
        # loudness cut-off
        loudness_threshold = 65.0 # dBA

        a = 1/(saturation_threshold-safe_threshold)
        b = -a*safe_threshold

        scores = (a*df['min_distance']+b)
        scores.name = 'safety'
        scores[df['min_distance'] >= saturation_threshold] = 1.0
        scores[df['loudness'] > loudness_threshold] = -0.5
        scores[df['unexpected_onsets'] > 0] = -1.0

        return scores

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

    def fit_gp(self, features, value, alpha, rbf_length= None, normalize= False):
        '''
        @param features: (n_samples, n_features)
        @param value: (n_samples,)
        @param alpha: observation noise level (std)
        @param rbf_length: length scale of RBF kernel

        @return: fitted GaussianProcessRegressor
        '''

        kernel = gp.kernels.ConstantKernel(1.0, constant_value_bounds="fixed")
        if rbf_length is not None:
            kernel*= gp.kernels.RBF(length_scale= rbf_length, length_scale_bounds="fixed")
        else:
            kernel*= gp.kernels.RBF()
        #kernel = gp.kernels.ConstantKernel(1.0)*gp.kernels.RBF()

        GPR= gp.GaussianProcessRegressor(
            n_restarts_optimizer=50,
            alpha=alpha**2,
            kernel= kernel,
            normalize_y= normalize,
            )
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

        nbp = RuckigPath.prototype(string= string, direction= direction)

        if len(plucks) < 1:
            rospy.logwarn(f"no plucks found for string {string} and finger {finger} in direction {direction}. returning default nbp as seed")
            nbp.string_position = actionspace.string_position[1]/2
            return nbp

        features = plucks[['string_position', 'keypoint_pos_y']].to_numpy()
        # features, features_norm_params = normalize(features)
        # use expected means/std instead:
        features_norm_params = (
            np.array([actionspace.string_position[1]/2, (actionspace.keypoint_pos_y[0] + (actionspace.keypoint_pos_y[1]-actionspace.keypoint_pos_y[0])/2) ]),
            np.array([actionspace.string_position[1]/4, (actionspace.keypoint_pos_y[1] - actionspace.keypoint_pos_y[0])/4])
        )
        features = normalize(features, features_norm_params)

        plucks['safety_score'] = self.score(plucks)

        # account for huge value span between successful and failed plucks with a low cutoff
        loudness_low_cutoff = 27.0 # dBA
        plucks['loudness'].fillna(loudness_low_cutoff, inplace= True)
        plucks[plucks['loudness'] < loudness_low_cutoff] = loudness_low_cutoff

        gp_loudness= self.fit_gp(features, plucks['loudness'], normalize= True, alpha= 2.0, rbf_length= 0.6)
        gp_safety= self.fit_gp(features, plucks['safety_score'], alpha= 0.05, rbf_length= 0.4)

        # limits are always given as lower(closer to pre), higher(closer to post), so invert if needed
        pos_limits= actionspace.keypoint_pos_y
        if direction < 0.0:
            pos_limits= -1.0*pos_limits[::-1]

        # maximize entropy
        def H(X):
            # X.reshape(1, -1)
            X= normalize(X, features_norm_params)
            return gp_loudness.predict(X, return_std=True)[1]

        # Probability of sample being safe w.r.t. Gaussian prediction
        def p_safe(X):
            # x= x.reshape(1, -1)
            X= normalize(X, features_norm_params)
            safety_score_predictions = gp_safety.predict(X , return_std=True)
            safety_prob = 1-stats.norm.cdf(0.0, *safety_score_predictions)
            return safety_prob

        domains = (
            (actionspace.string_position[0], actionspace.string_position[1]),
            (pos_limits[0], pos_limits[1]),
        )

        # optuna? In practice the sampling suffices and provides useful visualizations
        sample_size= 1000
        while True:
            # draw safe samples in F_domains

            # deterministic sampling, but different for each attempt
            rnd = np.random.default_rng(37+sample_size+len(plucks))
            samples = rnd.uniform(0, 1, size=(sample_size, len(domains)))
            samples = np.array([p*(d[1]-d[0])+d[0] for (p,d) in zip(samples.T, domains)]).T

            sample_H = H(samples)
            sample_psafe = p_safe(samples)

            safe_sample_cnt = np.sum(sample_psafe >= 0.95)
            if safe_sample_cnt > 20:
                break

            sample_size*= 2
            rospy.logerr(f"found too few safe samples (only {safe_sample_cnt}), will retry with {sample_size} samples")
            rospy.loginfo(f"knows {np.sum(plucks['safety_score'] > 0.0)} samples with safety_score > 0.0 and {np.sum(plucks['safety_score'] < 0.0)} samples with safety_score < 0.0")

        # limit to safe samples
        samples = samples[sample_psafe >= 0.95]
        sample_H = sample_H[sample_psafe >= 0.95]

        sample_max_H = samples[np.argmax(sample_H)]
        nbp.string_position= sample_max_H[0]
        nbp.keypoint_pos[0]= sample_max_H[1]
        rospy.loginfo(f"selected nbp: {nbp}")

        ## optional visualizations

        # safety scores of all considered samples
        fig, ax = plt.subplots(dpi= 100)
        cmap = sns.color_palette("seismic", as_cmap=True)
        norm = plt.Normalize(vmin=-1.0, vmax=1.0)
        sns.scatterplot(x= plucks['string_position'], y= plucks['keypoint_pos_y'], hue= plucks['safety_score'], palette= cmap, hue_norm= norm, legend= False, s= 100, ax= ax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=ax)
        ax.set_xlim(actionspace.string_position[0], actionspace.string_position[1])
        ax.set_ylim(pos_limits[0], pos_limits[1])
        publish_figure("episodes_safety_score", fig)

        # loudness of all known samples
        fig, ax = plt.subplots(dpi= 100)
        cmap = sns.color_palette("RdPu", as_cmap=True)
        norm = plt.Normalize(vmin=plucks['loudness'].min(), vmax=plucks['loudness'].max())
        if len(plucks['loudness']) > 1:
            sns.scatterplot(x= plucks['string_position'], y= plucks['keypoint_pos_y'], hue= plucks['loudness'], hue_norm= norm, palette= cmap, legend= False, s= 100, ax= ax)
        sm = plt.cm.ScalarMappable(norm= norm, cmap=cmap)
        fig.colorbar(sm, ax=ax)
        ax.set_xlim(actionspace.string_position[0], actionspace.string_position[1])
        ax.set_ylim(pos_limits[0], pos_limits[1])
        publish_figure("episodes_loudness", fig)

        # all safe samples evaluated for nbp
        fig, ax = plt.subplots(dpi= 100)
        cmap = sns.color_palette("RdPu", as_cmap=True)
        norm = plt.Normalize(vmin=sample_H.min(), vmax=sample_H.max())
        sns.scatterplot(x= samples[:,0], y= samples[:,1], hue= sample_H, hue_norm= norm, palette= cmap, legend= False, s= 100, ax= ax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm= norm)
        fig.colorbar(sm, ax=ax)
        ax.set_xlim(actionspace.string_position[0], actionspace.string_position[1])
        ax.set_ylim(pos_limits[0], pos_limits[1])
        publish_figure("sample_H", fig)

        # evaluate GP_loudness and p(safe|X) on a grid
        grid_size = 50
        grid_string_position = np.linspace(actionspace.string_position[0], actionspace.string_position[1], grid_size)
        grid_keypoint_pos_y = np.linspace(pos_limits[0], pos_limits[1], grid_size)
        xi, yi = np.meshgrid(
            grid_string_position,
            grid_keypoint_pos_y
            )
        grid_points = np.column_stack((xi.ravel(), yi.ravel()))
        means, std = gp_loudness.predict(normalize(grid_points, features_norm_params), return_std=True)

        # mask out points that are not safe
        means[p_safe(grid_points) < 0.95] = np.nan

        sample_psafe = p_safe(grid_points).reshape((grid_size, grid_size))

        def grid_plot(fig, values, cmap):
            values= values.reshape((grid_size, grid_size))
            ax= fig.gca()
            im= ax.imshow(values, origin='lower', cmap=cmap)
            plt.colorbar(im)
            ax.grid(False)
            ax.set_xticks(range(len(grid_string_position)), labels= [(f"{p:.2F}" if i%10==0 else "") for (i,p) in enumerate(grid_string_position)])
            ax.set_yticks(range(len(grid_keypoint_pos_y)), labels= [(f"{p:.4F}" if i%10==0 else "") for (i,p) in enumerate(grid_keypoint_pos_y)])
            ax.set_xlabel("string position")
            ax.set_ylabel("keypoint pos y")

        fig = plt.figure(dpi= 50)
        grid_plot(fig, means, plt.get_cmap("RdPu"))
        publish_figure("gp_loudness", fig)

        fig = plt.figure(dpi= 50)
        grid_plot(fig, std, plt.get_cmap("RdPu"))
        publish_figure("gp_std_loudness", fig)

        fig = plt.figure(dpi= 100)
        grid_plot(fig, sample_psafe, plt.get_cmap("brg"))
        publish_figure("p_safety", fig)

        return nbp

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

        GPR = self.fit_gp(features, loudnesses, alpha= 0.5, rbf_length= 1.0)

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
