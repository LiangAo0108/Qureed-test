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



class FourModeInspector(GenericDevice):
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
        
        "input3": Port(
            label="input3",
            direction="input",
            signal=None,
            signal_type=qureed.signals.generic_quantum_signal.GenericQuantumSignal,
            device=None),
        
        "input4": Port(
            label="input4",
            direction="input",
            signal=None,
            signal_type=qureed.signals.generic_quantum_signal.GenericQuantumSignal,
            device=None),
        
        
    }

    properties = {
	
    }
    

    gui_icon = HISTOGRAM

    gui_name = "4 mode inspector"
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
            qin2 = signals.get("input3", None)
            qin3 = signals.get("input4", None)
            self.log_message(f"inspector receives, {qin0} {qin1} {qin2} {qin3}")

            env0 = Envelope() if qin0 is None else qin0.contents
            env1 = Envelope() if qin1 is None else qin1.contents
            env2 = Envelope() if qin2 is None else qin2.contents
            env3 = Envelope() if qin3 is None else qin3.contents

            ce = CompositeEnvelope(env0, env1, env2, env3)
            ce.combine(env0.fock, env1.fock, env2.fock, env3.fock)

            # Force the representation into the matrix state
            env0.expand()

            state = ce.trace_out(env0.fock, env1.fock, env2.fock, env3.fock)
            self.log_state(f"inspector received a state:", state)
            self.log_message(f"env0: {env0.fock}, env1: {env1.fock}, env2: {env2.fock}, env3: {env3.fock}")

            self.log_state(f"ce.states[0]", ce.states[0].state)


            # Extract the probabilities
            dim0 = env0.fock.dimensions
            dim1 = env1.fock.dimensions
            dim2 = env2.fock.dimensions
            dim3 = env3.fock.dimensions
            self.log_message(f"env0.fock.dimensions = {env0.fock.dimensions}")
            self.log_message(f"env1.fock.dimensions = {env1.fock.dimensions}")
            self.log_message(f"env2.fock.dimensions = {env2.fock.dimensions}")
            self.log_message(f"env3.fock.dimensions = {env3.fock.dimensions}")
            unique_probabilities = {}
            for i in range(dim0):
                for j in range(dim1):
                    for k in range(dim2):
                        for l in range(dim3):

                            prob = state[((i * dim1 + j) * dim2 + k) * dim3 + l, ((i * dim1 + j) * dim2 + k) * dim3 + l]
                            if abs(prob) > 1e-5: # Filter out near-zero values
                                key = f"|{i},{j},{k},{l}>"

                                unique_probabilities[key] = jnp.abs(prob)
            filtered_probabilities = [f"{state} -> {prob}" for state, prob in unique_probabilities.items()]
            log_string = "\n".join(filtered_probabilities)
            self.log_message(f"\n{self.name} -> OUTPUT PROBABILITIES: \n" + log_string)
            self.log_message(f"State shape: {state.shape}")

        except Exception as e:
            self.log_message(traceback.format_exc())


    @log_action
    @schedule_next_event
    def des_action(self, time=None, *args, **kwargs):
        """
        Or implement this if you are implementing a trigger
        """
        pass
