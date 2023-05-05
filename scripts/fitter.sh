#!/bin/sh

if [ "$1" = "on" ]; then
    rosservice call /guzheng/string_fitter/set_active "data: true"
elif [ "$1" = "off" ]; then
    rosservice call /guzheng/string_fitter/set_active "data: false"
else
    echo "Usage: $0 <on/off>"
    exit 1
fi
