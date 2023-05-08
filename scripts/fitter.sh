#!/bin/sh

if [ "$1" = "on" ]; then
    rosservice call /guzheng/string_fitter/set_active "data: true"
elif [ "$1" = "off" ]; then
    rosservice call /guzheng/string_fitter/set_active "data: false"
elif [ "$1" = "load" ]; then
    rosservice call /guzheng/string_fitter/load_from_file
elif [ "$1" = "store" ]; then
    rosservice call /guzheng/string_fitter/store_to_file
else
    echo "Usage: $0 <on/off>"
    exit 1
fi
