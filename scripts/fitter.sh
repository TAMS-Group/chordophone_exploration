#!/bin/sh

if [ "$1" = "on" ]; then
    rosservice call /guzheng/string_fitter/set_active "data: true"
elif [ "$1" = "off" ]; then
    rosservice call /guzheng/string_fitter/set_active "data: false"
elif [ "$1" = "load" ]; then
    rosservice call /guzheng/string_fitter/load_from_file
elif [ "$1" = "store" ]; then
    rosservice call /guzheng/string_fitter/store_to_file
elif [ "$1" = "drop" ]; then
    rosservice call /guzheng/onset_projector/drop_events "name: '$2'"
else
    echo "usage: $0 <on/off/store/load>"
    echo "   in order to turn on/off the fitter"
    echo "   or to store/load fit strings"
    echo
    echo "or usage: $0 drop <note to drop>"
    echo "or usage: $0 drop ALL"
    echo "   in order to drop projected events"
    exit 1
fi
