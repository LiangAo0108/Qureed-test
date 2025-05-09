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


import custom



class CustomDelay(GenericDevice):
    ports = {
        
        "in": Port(
            label="in",
            direction="input",
            signal=None,
            signal_type=qureed.signals.generic_quantum_signal.GenericQuantumSignal,
            device=None),
        
        
        "out": Port(
            label="out",
            direction="output",  # Changed from "input" to "output" for clarity
            signal=None,
            signal_type=qureed.signals.generic_quantum_signal.GenericQuantumSignal,
            device=None),
        
    }

    properties = {
	
	"delay": {
	    "type": "float",
	}
	
    }
    

    gui_icon = FIBER

    gui_name = "Custom Delay"
    gui_tags = [
    "",
    
    ]
    gui_documentation = None
    
    power_peak = 0
    power_average = 0
    
    reference = None
      
    @log_action
    @schedule_next_event
    def des(self, time, *args, **kwargs):
        """
        Implement to use discrete event simulation
        """
        signal = kwargs["signals"]["in"]
        t = self.properties["delay"].get("value", 0.001)
        result = [("out", signal, time + t)]
        return result

    @log_action
    @schedule_next_event
    def des_action(self, time=None, *args, **kwargs):
        """
        Or implement this if you are implementing a trigger
        """
        pass
