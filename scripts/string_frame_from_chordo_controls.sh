#!/bin/bash

if [[ -z "$2" ]]; then
  echo "usage: $0 <chordo_controls-user-frame-number> <note-name>"
  exit 1
fi

rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 chordo_controls/user_defined_$1 guzheng/$2/head
