#!/usr/bin/env python

# breaks for episodes because some topics do not have headers!

import rosbag
import re
import sys

input_bag = sys.argv[1]

m = re.match('(.*)\.bag', input_bag)

output_bag = m.group(1) + '_rewritten_timestamps.bag'

# stolen from
# http://library.isr.ist.utl.pt/docs/roswiki/rosbag(2f)Cookbook.html

with rosbag.Bag(output_bag, 'w') as outbag:
    for topic, msg, t in rosbag.Bag(input_bag).read_messages():
        if topic == "/tf" and msg.transforms:
            outbag.write(topic, msg, msg.transforms[0].header.stamp)
        else:
            outbag.write(topic, msg, msg.header.stamp if msg._has_header else t)
