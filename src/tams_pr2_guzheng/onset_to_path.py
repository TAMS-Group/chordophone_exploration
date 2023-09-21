import copy
import librosa
import numpy as np
import os
import pandas as pd
import rospy
import sklearn.gaussian_process as gp
import scipy.stats as stats
import tams_pr2_guzheng.utils as utils

from nav_msgs.msg import Path
from tams_pr2_guzheng.msg import RunEpisodeResult
from tams_pr2_guzheng.paths import RuckigPath
from tams_pr2_guzheng.utils import normalize
from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns

class OnsetToPath:
    def __init__(self, *, storage : str = '/tmp/plucks.json'):
        self.pluck_table = pd.DataFrame(
            columns=(*RuckigPath().params_map.keys(), 'finger', 'loudness', 'detected_note', 'onset_cnt', 'onsets')
            )
        self.storage = storage
        if os.path.exists(self.storage):
            self.pluck_table = pd.read_json(self.storage)
            self.pluck_table['safety_score'] = utils.score_safety(self.pluck_table)
        self.print_summary()

        rospy.on_shutdown(self.store_plucks)

    def store_plucks(self):
        rospy.loginfo(f"storing plucks in '{self.storage}'")
        self.store_to_file()

    def print_summary(self):
        summary= f"OnsetToPath stores {len(self.pluck_table)} plucks\n"
        for n in self.pluck_table['detected_note'].unique():
            if isinstance(n, float) and np.isnan(n):
                l= len(self.pluck_table[self.pluck_table['detected_note'].isna()])
            else:
                l= len(self.pluck_table[self.pluck_table['detected_note'] == n])
            summary+= f"{n}: {l} plucks\n"
        summary+= "\n"

        for n in set(self.pluck_table['string']):
            summary+= f"{n}: {len(self.pluck_table[self.pluck_table['string'] == n])} plucks\n"

        rospy.loginfo(summary)

    def store_to_file(self):
        # exclude safety_score from storage, we compute it on load
        self.pluck_table.drop(columns= ['safety_score']).to_json(self.storage)

    def add_sample(self, row):
        row['onsets'] = str(row['onsets']) # convert to string for eventual json serialization
        row_df = pd.DataFrame(row, columns= row.keys(), index= [0])
        row_df['safety_score'] = utils.score_safety(row_df)
        self.pluck_table = pd.concat((self.pluck_table, row_df), axis= 0, ignore_index=True)

        self.plot_loudness_strips()

    def score_row(self, row):
        r = copy.deepcopy(row)
        try:
            del r['onsets'] # this field is a list of onsets, but pandas just ignores the whole dict if it sees it
        except KeyError:
            pass
        df = pd.DataFrame(r, columns= r.keys(), index= [0])

        return utils.score_safety(df)[0]

    def plot_loudness_strips(self):
        '''plot cross-string loudness overview'''

        # we want to keep the original order in pluck_table to notice temporal effects
        # so we prepare a copy for plotting
        X = self.pluck_table[self.pluck_table['safety_score'] > 0].copy()
        X= X.sort_values('string', key= lambda x: x.map(lambda a: librosa.note_to_midi(utils.string_to_note(a))))
        X['direction'] = self.pluck_table['pre_y'].map(lambda y: 'inwards' if y < 0.0 else 'outwards')

        X['loudness'] = X['loudness'].fillna(-5.0)

        backend= plt.get_backend()
        plt.switch_backend('agg')
        fig = plt.figure(dpi= 100)
        ax : plt.Axes = sns.stripplot(x=X['string'], y=X['loudness'], hue= X['direction'], hue_order= ['inwards', 'outwards'], ax = fig.gca())
        ax.set_ylabel('loudness [dBA]')
        utils.publish_figure("loudness_strips", fig)
        plt.switch_backend(backend)

    def infer_next_best_pluck(self, *, string : str, finger : str, actionspace : RuckigPath.ActionSpace, direction : float) -> RuckigPath:
        assert(direction in [-1.0, 1.0])

        relevant_plucks = self.pluck_table[
            (self.pluck_table['string'] == string) &
            (self.pluck_table['finger'] == finger) &
            (self.pluck_table['post_y']*direction >= 0.0)
        ]

        plucks = relevant_plucks[actionspace.is_valid(relevant_plucks)].copy()

        if relevant_plucks.size != plucks.size:
            rospy.logwarn(f'dropped {relevant_plucks.size - plucks.size} plucks invalid w.r.t. passed actionspace')

        nbp = RuckigPath.prototype(string= string, direction= direction)
        nbp.string_position = actionspace.string_position[1]/2

        if len(plucks) < 1:
            rospy.logwarn(f"no plucks found for string {string} and finger {finger} in direction {direction}. returning default nbp as seed")
            return nbp

        features = plucks[['string_position', 'keypoint_pos_y']]
        # features, features_norm_params = normalize(features)
        # use expected means/std instead:
        features_norm_params = (
            np.array([actionspace.string_position[1]/2, (actionspace.keypoint_pos_y[0] + (actionspace.keypoint_pos_y[1]-actionspace.keypoint_pos_y[0])/2) ]),
            np.array([actionspace.string_position[1]/4, (actionspace.keypoint_pos_y[1] - actionspace.keypoint_pos_y[0])/4])
        )
        features = normalize(features, features_norm_params).values

        if not (plucks['safety_score'] > 0).any():
            rospy.logwarn(f"no safe plucks found for string {string} and finger {finger} in direction {direction}. dropping unsafe plucks and return default nbp as seed")
            self.pluck_table.drop(plucks.index, inplace= True)
            return nbp

        # account for huge value span between successful and failed plucks with a low cutoff
        loudness_low_cutoff = 15.0 # dBA
        plucks['loudness'].fillna(loudness_low_cutoff, inplace= True)
        plucks.loc[plucks['loudness'] < loudness_low_cutoff, 'loudness'] = loudness_low_cutoff

        gp_loudness= utils.fit_gp(
            features,
            plucks['loudness'].values,
            normalize= True,
            alpha= 1.0,
            rbf_length= (0.5, 0.25),
            train= False # True if plucks['loudness'].dropna().size > 50 else False
        )
        gp_safety = utils.fit_gp(
            features,
            plucks['safety_score'].values,
            normalize= False,
            alpha= .5,
            rbf_length= (0.5, 0.25),
            train= False
        )

        # maximize entropy
        def H(X):
            # X.reshape(1, -1)
            X= normalize(X, features_norm_params)
            return gp_loudness.predict(X, return_std=True)[1]

        # Probability of sample being safe w.r.t. Gaussian prediction
        def p_safe(X):
            safety_score_predictions = gp_safety.predict(normalize(X, features_norm_params), return_std=True)
            return utils.prob_gt_zero(safety_score_predictions)

        domains = (
            actionspace.string_position,
            actionspace.keypoint_pos_y,
        )

        psafe_threshold = 0.7

        # optuna? In practice the sampling suffices and provides useful visualizations
        sample_size= 2000
        while (not rospy.is_shutdown()):
            if sample_size > 1e6:
                rospy.logerr(f"could not find a safe sample for {string} in direction {direction} after {float(sample_size):.0e} attempts. restarting with default nbp as seed")
                self.pluck_table.drop(plucks.index, inplace= True)
                return nbp

            # deterministic sampling, but different for each attempt
            rnd = np.random.default_rng(37+sample_size+len(plucks))

            # sample in domains
            samples = rnd.uniform(0, 1, size=(sample_size, len(domains)))
            samples = np.array([p*(d[1]-d[0])+d[0] for (p,d) in zip(samples.T, domains)]).T

            samples_H = H(samples)
            samples_psafe = p_safe(samples)

            safe_sample_cnt = np.sum(samples_psafe >= psafe_threshold)
            if safe_sample_cnt > 20:
                break
            sample_size*= 2
            rospy.loginfo(f"found too few safe samples (only {safe_sample_cnt}), will retry with {sample_size} samples")
            rospy.loginfo_throttle(10, f"knows {np.sum(plucks['safety_score'] > 0.0)} samples with safety_score > 0.0 and {np.sum(plucks['safety_score'] < 0.0)} samples with safety_score < 0.0")

        # limit to sufficiently safe samples
        indices = samples_psafe >= psafe_threshold
        samples = samples[indices]
        samples_H = samples_H[indices]

        sample_max_H = samples[np.argmax(samples_H)]
        nbp.string_position= sample_max_H[0]
        nbp.keypoint_pos[0]= sample_max_H[1]
        rospy.loginfo(f"selected nbp: {nbp} with H= {np.max(samples_H)} out of {len(samples)} sufficiently safe samples")

        ## optional visualizations

        # safety scores of all considered trials
        fig, ax = plt.subplots(dpi= 100)
        utils.plot_trials(
            plucks,
            col= plucks['safety_score'],
            cmap= sns.color_palette("seismic", as_cmap=True),
            norm= plt.Normalize(vmin=-1.0, vmax=1.0),
            actionspace= actionspace,
            ax= ax,
        )
        utils.publish_figure("episodes_safety_score", fig)

        # loudness of all known samples
        fig, ax = plt.subplots(dpi= 100)
        utils.plot_trials(
            plucks,
            col= plucks['loudness'],
            cmap= sns.color_palette("RdPu", as_cmap=True),
            nan= 'green',
            nan_label= 'miss',
            actionspace= actionspace,
            ax= ax,
        )
        utils.publish_figure("episodes_loudness", fig)

        fig, ax = plt.subplots(dpi= 100)
        cmap= sns.color_palette('icefire', as_cmap= True)
        spread = max((abs(plucks['keypoint_pos_y'].min()), abs(plucks['keypoint_pos_y'].max())))
        norm = plt.Normalize(-spread, spread)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        art = sns.scatterplot(x='string_position', y='loudness', data=plucks, hue='keypoint_pos_y', hue_norm=norm, palette=cmap, legend=False, ax= ax)
        art.set_title('loudness along string')
        ax.set_xlim(*actionspace.string_position)
        art.figure.colorbar(sm, ax=art)
        utils.publish_figure("episodes_loudness_along_string", fig)

        df_samples = pd.DataFrame({'string_position' : samples[:,0], 'keypoint_pos_y' : samples[:,1], 'H': samples_H})
        # all safe samples evaluated for nbp
        fig, ax = plt.subplots(dpi= 100)
        utils.plot_trials(
            df_samples,
            col= df_samples['H'],
            cmap= sns.color_palette("RdPu", as_cmap=True),
            actionspace= actionspace,
            ax= ax,
        )
        ax.scatter(sample_max_H[0], sample_max_H[1], marker= 'x', color= 'blue', s= 150, linewidths= 5)
        utils.publish_figure("sample_H", fig)

        # evaluate GP_loudness and p(safe|X) on a grid
        grid_size= 50
        grid_points = utils.make_grid_points(actionspace, 50).values
        means, std = gp_loudness.predict(normalize(grid_points, features_norm_params), return_std=True)

        # mask out points that are not safe
        means[p_safe(grid_points) < psafe_threshold] = np.nan

        p_safe = p_safe(grid_points).reshape((grid_size, grid_size))

        fig = plt.figure(dpi= 50)
        utils.grid_plot(means, actionspace, plt.get_cmap("RdPu"))
        plt.title(f"GP safe loudness mean for {string} in direction {direction} with finger {finger}")

        utils.publish_figure("gp_loudness", fig)

        fig = plt.figure(dpi= 50)
        utils.grid_plot(std, actionspace, plt.get_cmap("RdPu"))
        utils.publish_figure("gp_std_loudness", fig)

        fig = plt.figure(dpi= 100)
        utils.grid_plot(p_safe, actionspace, plt.get_cmap("brg"))
        utils.publish_figure("p_safety", fig)

        return nbp

    def get_note_min_max(self, note : str, direction : float = 0.0):
        '''
        Returns the minimum and maximum loudness for the given note.
        '''
        plucks = self.pluck_table[
            (self.pluck_table['string'] == utils.note_to_string(note)) &
            (self.pluck_table['detected_note'] == note) &
            (self.pluck_table['onset_cnt'] == 1) &
            (self.pluck_table['safety_score'] > 0.0) &
            (self.pluck_table['post_y']*direction >= 0.0)
        ]
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
            (self.pluck_table['string'] == utils.note_to_string(note)) &
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

        string = utils.note_to_string(note)

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

        gp_loudness= utils.fit_gp(
            features[plucks['safety_score'] > 0],
            plucks.loc[plucks['safety_score'] > 0, 'loudness'].values,
            normalize= True,
            alpha= 1.0,
            rbf_length= (0.5, 0.25),
            train= False # True if plucks['loudness'].dropna().size > 50 else False
        )
        gp_safety = utils.fit_gp(
            features,
            plucks['safety_score'].values,
            normalize= False,
            alpha= .5,
            rbf_length= (0.5, 0.25),
            train= False
        )

        def DistanceToTargetLoudness(X):
            # X.reshape(1, -1)
            X= normalize(X, features_norm_params)
            return np.abs(gp_loudness.predict(X, return_std=False)-loudness)

        # Probability of sample being safe w.r.t. Gaussian prediction
        def p_safe(X):
            safety_score_predictions = gp_safety.predict(normalize(X, features_norm_params), return_std=True)
            return utils.prob_gt_zero(safety_score_predictions)

        domains = (
            (plucks['string_position'].min(), plucks['string_position'].max()) if string_position is None else (max((0.0, string_position-.05)), min((plucks['string_position'].max(), string_position+.05))),
            (plucks['keypoint_pos_y'].min(), plucks['keypoint_pos_y'].max()),
        )

        psafe_threshold = 0.9

        # optuna?
        sample_size= 5000

        # deterministic sampling, but different for each attempt
        rnd = np.random.default_rng(37+int(loudness*10)+int((0 if string_position is None else string_position)*1000)+len(plucks))

        # sample in domains
        samples = rnd.uniform(0, 1, size=(sample_size, len(domains)))
        samples = np.array([p*(d[1]-d[0])+d[0] for (p,d) in zip(samples.T, domains)]).T

        samples_distance = DistanceToTargetLoudness(samples)
        samples_psafe = p_safe(samples)

        # limit to sufficiently safe samples
        indices = samples_psafe >= psafe_threshold
        samples = samples[indices]
        samples_distance = samples_distance[indices]

        sample_closest = samples[np.argmin(samples_distance)]
        p = RuckigPath.prototype(string= string, direction= direction)
        p.string_position= sample_closest[0]
        p.keypoint_pos[0]= sample_closest[1]

        # TODO: maximize pdf between all samples around 1dB distance

        return p, finger, 1.0