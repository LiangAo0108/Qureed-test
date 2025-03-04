"""

 created with template
"""
from typing import Union
import qureed
from qureed.devices import (
    GenericDevice, schedule_next_event,
    coordinate_gui, log_action )
from qureed.assets.icon_list import *
from qureed.devices.port import Port
from qureed.signals import *
import numpy as np
from photon_weave.state.envelope import Envelope
from qureed.devices import (
    GenericDevice,
    coordinate_gui,
    log_action,
    schedule_next_event,
)
from qureed.devices.port import Port
from qureed.signals import (
    GenericBoolSignal,
    GenericIntSignal,
    GenericQuantumSignal,
    GenericSignal,
)

import custom



class DelayNphotonsource(GenericDevice):
    ports = {
        
        "trigger": Port(
            label="trigger",
            direction="input",
            signal=None,
            signal_type=qureed.signals.generic_bool_signal.GenericBoolSignal,
            device=None),
        
        "photon_num": Port(
            label="photon_num",
            direction="input",
            signal=None,
            signal_type=qureed.signals.generic_int_signal.GenericIntSignal,
            device=None),
        
        "delay": Port(
            label="delay",
            direction="input",
            signal=None,
            signal_type=qureed.signals.generic_float_signal.GenericFloatSignal,
            device=None),
        
        
        "output": Port(
            label="output",
            direction="output",  # Changed from "input" to "output" for clarity
            signal=None,
            signal_type=qureed.signals.generic_quantum_signal.GenericQuantumSignal,
            device=None),
        
    }

    properties = {
	
    }
    

    gui_icon = N_PHOTON_SOURCE

    gui_name = "Delay NPhotonSource"
    gui_tags = [
    "",
    
    ]
    gui_documentation = None
    power_peak = 0
    power_average = 0
    reference = None

    def __init__(self, name=None, time=0, uid=None, **kwargs):
        super().__init__(name=name, uid=uid)
        self.photon_num = None
        self.delay = 0.0

    #@ensure_output_compute
    @coordinate_gui
    #@wait_input_compute


    def set_photon_num(self, photon_num: int):
        """
        Set the number of photons the source should emit in a pulse
        """
        self.photon_num = photon_num
      
    @log_action
    @schedule_next_event
    def des(self, time, *args, **kwargs):
        """
        Implement to use discrete event simulation
        """
        if "photon_num" in kwargs["signals"]:
            self.photon_num=float(kwargs["signals"]["photon_num"].contents)

        elif "delay" in kwargs["signals"]:
            self.delay=float(kwargs["signals"]["delay"].contents)

        elif "trigger" in kwargs["signals"] and self.photon_num is not None:
            n = int(self.photon_num)
            # Creating new envelope
            env = Envelope()
            env.fock.state = n
            # Creating output
            signal = GenericQuantumSignal()
            signal.set_contents(content=env)
            delay_time = time + self.delay
            result = [("output", signal, delay_time)]
            return result
        else:
            raise Exception("Unknown Photon Num")


    @log_action
    @schedule_next_event
    def des_action(self, time=None, *args, **kwargs):
        """
        Or implement this if you are implementing a trigger
        """
        pass