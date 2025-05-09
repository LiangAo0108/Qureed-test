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

from photon_weave.state.envelope import Envelope
from photon_weave.photon_weave import Config


import custom



class CustomSource(GenericDevice):
    ports = {
        "trigger": Port(
            label="trigger",
            direction="input",
            signal=None,
            signal_type=GenericBoolSignal,
            device=None,
        ),
        
        "out": Port(
            label="out",
            direction="output",
            signal=None,
            signal_type=qureed.signals.generic_quantum_signal.GenericQuantumSignal,
            device=None),
        
    }

    properties = {
	
	"photonNum": {
	    "type": "int",
	},
	"dimensions": {
	    "type": "int",
	}
	
    }
    

    gui_icon = N_PHOTON_SOURCE

    gui_name = "Custom Source"
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
        C = Config()
        photon_num = int(self.properties["photonNum"].get("value", False))
        dimensions = int(self.properties["dimensions"].get("value", (photon_num+2)*4))
        C.set_contraction(True)
        if photon_num:
            env = Envelope()
            env.fock.state = photon_num
            env.fock.dimensions = dimensions
            env.fock.uid = f"{self.properties['name'].get('value', '')}-{env.fock.uid}"
            env.fock.expand()
            env.fock.expand()
            signal = GenericQuantumSignal()
            signal.set_contents(content=env)
            result = [("out", signal, time+1e-10)]
            return result



    @log_action
    @schedule_next_event
    def des_action(self, time=None, *args, **kwargs):
        """
        Or implement this if you are implementing a trigger
        """
        pass
