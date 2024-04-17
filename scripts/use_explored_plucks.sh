#!/bin/sh

PS4='$ '
set -x

cp $(rospack find tams_pr2_guzheng)/data/plucks_explore.json $(rospack find tams_pr2_guzheng)/data/plucks.json