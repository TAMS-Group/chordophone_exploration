#!/usr/bin/env python

import rospy

from threading import Lock

from tams_pr2_guzheng.msg import ActionParameters

import matplotlib.pyplot as plt

from itertools import combinations

class Plotter():
    def __init__(self, just_play=False):
        self.parameters= []
        self.lock = Lock()
         
        self.parameter_sub = rospy.Subscriber(
            'episode/action_parameters',
            ActionParameters,
            self.cb,
            queue_size=10,
            tcp_nodelay=True)

    def cb(self, msg):
        with self.lock:
            self.parameters.append(msg.action_parameters)

def main():
    rospy.init_node('plot_parameters')

    p= Plotter() 

    plt.ion()
    r= rospy.Rate(0.5)

    while not rospy.is_shutdown():
        r.sleep()

        idx= 1
        with p.lock:
            for i,j in combinations(range(4),2):
                plt.subplot(2,3, idx)
                idx+= 1
                plt.title(f'{i} / {j}')
                plt.scatter(
                    [pa[i] for pa in p.parameters],
                    [pa[j] for pa in p.parameters],
                    color= 'g')
        plt.show()
        plt.pause(0.1)



if __name__ == "__main__":
    main()
