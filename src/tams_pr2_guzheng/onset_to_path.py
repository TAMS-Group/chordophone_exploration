import copy
import numpy as np
import os
import pandas as pd
from nav_msgs.msg import Path
from .paths import RuckigPath
from .utils import note_to_string

class OnsetToPath:
    def __init__(self, storage = '/tmp/plucks.json'):
        self.pluck_table = pd.DataFrame(
            columns=(*RuckigPath().params_map.keys(), 'loudness', 'detected_note', 'onset_cnt', 'onsets')
            )
        self.storage = storage
        if os.path.exists(self.storage):
            self.pluck_table = pd.read_json(self.storage)

    def store_to_file(self):
        self.pluck_table.to_json(self.storage)

    def add_sample(self, row):
        row_df = pd.DataFrame(row, columns= row.keys(), index= [0])
        self.pluck_table = pd.concat((self.pluck_table, row_df), ignore_index=True)

    def get_path(self, note : str, loudness : float, direction = 0.0, string_position= None) -> Path:
        '''
        Returns a path for the given note and loudness.

        @param note: The note to play
        @param loudness: The loudness to play the note at
        @param direction: The direction to pluck in (1 for towards the robot, -1 for away from the robot)
        '''
        note_plucks = self.pluck_table[
            (self.pluck_table['note'] == note_to_string(note)) &
            (self.pluck_table['detected_note'] == note) &
            (self.pluck_table['onset_cnt'] == 1) &
            (self.pluck_table['post_y']*direction >= 0.0)]
        if len(note_plucks) == 0:
            raise ValueError(f"No plucks found for note {note} in direction {direction}")

        objective = np.abs(note_plucks['loudness'] - loudness)
        # if string_position is not None:
        #     objective+= 10*np.abs(note_plucks['string_position']-string_position)

        pluck = note_plucks.iloc[np.argmin(objective)]

        pluck= copy.deepcopy(pluck)
        pluck['string_position'] = string_position
        return RuckigPath.from_map(pluck)()
