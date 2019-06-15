#!/usr/bin/python3

########################################################################################################
# Author            : Aggelos Kolaitis <neoaggelos@gmail.com>
# Description       : Simulation program
# Requires          : python3, numpy
# Last Update       : 2019-06-15
# Usage             :
########################################################################################################

import numpy as np
from typing import Any
import logging

logging.basicConfig(level=logging.DEBUG)

####################################################################
# Random numbers

# to have same results
SEED = 1123412341
np.random.seed(SEED)

# Exp(S)
def exp(*args, **kwargs):
    return np.random.exponential(*args, **kwargs)

# Erlang-k(S) = sum_of_k_independent( Exp(S/k) )
def erlangk(S, k):
    return np.sum(exp(S/k, size=k))


####################################################################
# Utility

def die(msg):
    '''live with honor, die with glory'''
    print(msg)
    exit(-1)


def float_to_str(f):
    '''to string with two decimals'''
    return '%.2f' % f


####################################################################
# Parameters
PARAM_N = 75
PARAM_Z = 39
PARAM_A = 20
PARAM_B = 4
PARAM_S = 1.5
PARAM_k = 4


####################################################################
# State and initialization

class Event:
    '''describes an event'''
    def __init__(self, type, time, param=None):
        self.type = type
        self.time = time
        self.param = param

    def __str__(self):
        time = (float_to_str(self.time))
        return f'Event(time={time}, type={self.type}, param={self.param})'

    def __repr__(self):
        return self.__str__()


class Cycle:
    '''describes a completed cycle'''
    def __init__(self, response_times, duration):
        self.duration = duration
        self.response_times = response_times

        self.throughput = len(self.response_times) / self.duration

    def __str__(self):
        duration = float_to_str(self.duration)
        count = len(self.response_times)
        x = float_to_str(self.throughput)
        mean = float_to_str(np.average(self.response_times))
        return f'Cycle(R={mean}, X={x}, N={count}, duration={duration})'

    def __repr__(self):
        return self.__str__()


class Simulation:
    def __init__(self):
        # start time of current cycle
        self.cycle_start = 0

        # array of response times for current cycle
        self.response_times = []

        # server is working/not working
        self.server_up = True

        # time that server will break down and recover
        self.next_downtime = exp(PARAM_A)
        self.next_uptime = self.next_downtime + exp(PARAM_B)

        # next arrival times for each client
        self.next_arrival = exp(PARAM_Z, size=PARAM_N)

        # last arrival times for each client (needed for response time calculations)
        self.last_arrival = np.inf * np.ones(PARAM_N)

        # finish times for each client (will be initialized upon arrival)
        self.next_finish = np.inf * np.ones(PARAM_N)

        # number of active clients
        self.num_clients = 0

        # list of completed cycles (as Cycle objects)
        self.cycles = []


    def get_next_event(self):
        '''return next event'''
        min_arrival = np.min(self.next_arrival)
        min_finish = np.min(self.next_finish)

        next_event = min(min_arrival, min_finish, self.next_downtime, self.next_uptime)

        if next_event == self.next_downtime:
            # assert self.server_up == True
            return Event(type='down', time=self.next_downtime)

        elif next_event == self.next_uptime:
            # assert self.server_up == False
            return Event(type='up', time=self.next_uptime)

        elif next_event == min_arrival:
            return Event(type='arrival', time=min_arrival, param=np.argmin(self.next_arrival))

        elif next_event == min_finish:
            # assert self.server_up == True
            return Event(type='finish', time=min_finish, param=np.argmin(self.next_finish))

        else:
            die('Run awaaaaaaaaaaay, be freeeeeee!')


    def handle_event(self, e=None):
        '''handle a specific event'''

        if e is None:
            e = self.get_next_event()

        # logging.info(e)
        # assert e.time < np.inf

        if e.type == 'arrival':
            # time of finish
            start = e.time if self.server_up else self.next_uptime

            D = erlangk(PARAM_S, PARAM_k)
            self.next_finish[e.param] = start + D
            self.next_arrival[e.param] = np.inf
            self.last_arrival[e.param] = e.time

            # one more client
            self.num_clients += 1

        elif e.type == 'down':
            # delay all currently connected clients
            self.next_finish += self.next_uptime - self.next_downtime
            self.next_downtime = np.inf
            self.server_up = False

        elif e.type == 'up':
            self.server_up = True

            self.next_downtime = e.time + exp(PARAM_A)
            self.next_uptime = self.next_downtime + exp(PARAM_B)

            if self.num_clients == 0 and self.response_times:
                cycle = Cycle(duration=e.time - self.cycle_start,
                              response_times=self.response_times)

                logging.info(cycle)

                self.cycles.append(cycle)
                self.response_times = []
                self.cycle_start = e.time

        elif e.type == 'finish':
            # assert self.server_up == True
            # assert self.last_arrival[e.param] is not np.inf

            # calculate response time
            self.response_times.append(e.time - self.last_arrival[e.param])

            # one less client
            self.num_clients -= 1

            # plan next arrival
            self.next_arrival[e.param] = e.time + exp(PARAM_Z)
            self.next_finish[e.param] = np.inf

s = Simulation()

for _ in range(500000):
    s.handle_event()

logging.info(f'Completed {len(s.cycles)} cycles')