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
import traceback
from photon_weave.state.envelope import Envelope
from photon_weave.state.composite_envelope import CompositeEnvelope
import jax.numpy as jnp



class Inspector(GenericDevice):
    ports = {
        
        "input1": Port(
            label="input1",
            direction="input",
            signal=None,
            signal_type=qureed.signals.generic_quantum_signal.GenericQuantumSignal,
            device=None),
        
        "input2": Port(
            label="input2",
            direction="input",
            signal=None,
            signal_type=qureed.signals.generic_quantum_signal.GenericQuantumSignal,
            device=None),
        
        
    }

    properties = {
	
    }
    

    gui_icon = HISTOGRAM

    gui_name = "Inspector"
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
        try:
            signals = kwargs.get("signals", {})

            qin0 = signals.get("input1", None)
            qin1 = signals.get("input2", None)
            self.log_message(f"inspector receives, {qin0} {qin1}")

            env0 = Envelope() if qin0 is None else qin0.contents
            env1 = Envelope() if qin1 is None else qin1.contents

            ce = CompositeEnvelope(env0, env1)
            ce.combine(env0.fock, env1.fock)

            # Force the representation into the matrix state
            env0.expand()

            state = ce.trace_out(env0.fock, env1.fock)
            self.log_state(f"inspector received a state:", state)

            # Extract the probabilities
            dim0 = env0.fock.dimensions
            dim1 = env1.fock.dimensions

            unique_probabilities = {}
            for i in range(dim0):
                for j in range(dim1):
                    prob = state[i * dim1 + j, i * dim1 + j]
                    if abs(prob) > 1e-10:  # Filter out near-zero values
                        unique_probabilities[f"|{i},{j}>"] = jnp.abs(prob)
            filtered_probabilities = [f"{state} -> {prob}" for state, prob in unique_probabilities.items()]
            log_string = "\n".join(filtered_probabilities)
            self.log_message(f"\n{self.name} -> OUTPUT PROBABILITIES: \n" + log_string)

        except Exception as e:
            traceback.print_exc()



    @log_action
    @schedule_next_event
    def des_action(self, time=None, *args, **kwargs):
        """
        Or implement this if you are implementing a trigger
        """
        pass