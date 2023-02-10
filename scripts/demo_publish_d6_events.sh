#!/bin/sh

rostopic pub -s -r ${1:-1} /guzheng/onsets tams_pr2_guzheng/NoteOnset "header:
  seq: 0
  stamp: now
  frame_id: 'guzheng'
note: 'd6'
confidence: 1.0"
