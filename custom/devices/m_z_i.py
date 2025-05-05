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
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.operation import Operation, CompositeOperationType
from photon_weave.operation import FockOperationType
import jax.numpy as jnp
import custom
import traceback



class Mzi(GenericDevice):
    ports = {
        
        "qin0": Port(
            label="qin0",
            direction="input",
            signal=None,
            signal_type=qureed.signals.generic_quantum_signal.GenericQuantumSignal,
            device=None),
        
        "qin1": Port(
            label="qin1",
            direction="input",
            signal=None,
            signal_type=qureed.signals.generic_quantum_signal.GenericQuantumSignal,
            device=None),
        
        "phase_shift": Port(
            label="phase_shift",
            direction="input",
            signal=None,
            signal_type=qureed.signals.generic_float_signal.GenericFloatSignal,
            device=None),
        
        
        "qout0": Port(
            label="qout0",
            direction="output",  # Changed from "input" to "output" for clarity
            signal=None,
            signal_type=qureed.signals.generic_quantum_signal.GenericQuantumSignal,
            device=None),
        
        "qout1": Port(
            label="qout1",
            direction="output",  # Changed from "input" to "output" for clarity
            signal=None,
            signal_type=qureed.signals.generic_quantum_signal.GenericQuantumSignal,
            device=None),
        
    }

    properties = {
	
    }
    

    gui_icon = HISTOGRAM

    gui_name = "MZI"
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

        try:
            if "phase_shift" in kwargs.get("signals"):
                self.phase_shift = kwargs["signals"]["phase_shift"].contents

            signals = kwargs.get("signals", {})

            # get the envelopes
            qin0_signal = signals.get("qin0", None)
            qin1_signal = signals.get("qin1", None)

            if qin0_signal is None:
                env0 = Envelope()
            else:
                env0 = signals["qin0"].contents

            if qin1_signal is None:
                env1 = Envelope()
            else:
                env1 = signals["qin1"].contents

            print(f"qin0_signal: {qin0_signal}, qin1_signal: {qin1_signal}")
            self.log_message(f"env0: {env0}, env1: {env1}")

            if (qin0_signal is None) and (qin1_signal is None):
                print("MZI has no input")
                self.log_message("MZI no input here")
                return

            ce = CompositeEnvelope(env0, env1)
            ce.combine(env0.fock, env1.fock)
            # print (f"ce.states[0]:{ce.states[0]}")

            # create operator for beam splitter and phase shifter
            fo_beam_splitter = Operation(CompositeOperationType.NonPolarizingBeamSplitter, eta=jnp.pi / 4)
            fo_phase_shifter = Operation(FockOperationType.PhaseShift, phi=self.phase_shift)

            # implement a MZI with beam splitter, phase shifter, beam splitter
            # apply the beam splitter
            ce.apply_operation(fo_beam_splitter, env0.fock, env1.fock)
            print(f"ce.states[0] after first beam splitter: {ce.states[0]}")
            self.log_state(f"ce.states[0] after first beam splitter: ", ce.states[0].state)

            # apply the phase shifter
            ce.apply_operation(fo_phase_shifter, env1.fock)
            # apply the beam splitter twice
            ce.apply_operation(fo_beam_splitter, env0.fock, env1.fock)

            # create new quantum signals
            qout0_signal = GenericQuantumSignal()
            qout1_signal = GenericQuantumSignal()

            # set the modified envelope to the content of output signal
            qout0_signal.set_contents(env0)
            qout1_signal.set_contents(env1)

            # return to the result
            result = [("qout0", qout0_signal, time), ("qout1", qout1_signal, time)]
            self.log_message(f"Unit Cell returning {kwargs['signals']}")
            return result

        except Exception as e:
            self.log_message(traceback.format_exc())


    @log_action
    @schedule_next_event
    def des_action(self, time=None, *args, **kwargs):
        """
        Or implement this if you are implementing a trigger
        """
        pass