"""

 created with template
"""
from typing import Union
import jax.numpy as jnp
import qureed
from qureed.devices import (
    GenericDevice, schedule_next_event,
    coordinate_gui, log_action )
from qureed.assets.icon_list import *
from qureed.devices.port import Port
from qureed.signals import *
from photon_weave.state.envelope import Envelope
from photon_weave.state.fock import Fock
from photon_weave.state.composite_envelope import CompositeEnvelope
from photon_weave.operation import Operation, CompositeOperationType
from photon_weave.operation import FockOperationType
from photon_weave._math.ops import annihilation_operator, creation_operator
import jax.numpy as jnp
import custom
import traceback

    


class UnitCell(GenericDevice):
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
        
        "external_phase_shift": Port(
            label="external_phase_shift",
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
    

    gui_icon = FUNNEL

    gui_name = "Unit Cell"
    gui_tags = [
    "",
    
    ]
    gui_documentation = None
    
    power_peak = 0
    power_average = 0
    
    reference = None

    def __init__(self, name=None, uid=None, **kwargs):  # initialize MZI including device name and unique identifier
        super().__init__(name=name, uid=uid)
        self.phase_shift = 0
        self.external_phase_shift = 0

    @log_action
    @schedule_next_event
    def des(self, time, *args, **kwargs):
        """
        Implement to use discrete event simulation
        """
        try:
            if "phase_shift" in kwargs.get("signals"):
                self.phase_shift = kwargs["signals"]["phase_shift"].contents
            if "external_phase_shift" in kwargs.get("signals"):
                self.external_phase_shift = kwargs["signals"]["external_phase_shift"].contents

            signals = kwargs.get("signals", {})

            # get the envelopes
            qin0_signal = signals.get("qin0", None)
            qin1_signal = signals.get("qin1", None)

            if (qin0_signal is None) and (qin1_signal is None):
                return

            if qin0_signal is None:
                env0 = Envelope()
            else:
                env0 = signals["qin0"].contents

            if qin1_signal is None:
                env1 = Envelope()
            else:
                env1 = signals["qin1"].contents

            #print(f"qin0_signal: {qin0_signal}, qin1_signal: {qin1_signal}")
            #self.log_message(f"env0: {env0}, env1: {env1}")


            ce = CompositeEnvelope(env0, env1)
            ce.combine(env0.fock, env1.fock)

            # create operator for beam splitter and phase shifter
            fo_beam_splitter = Operation(CompositeOperationType.NonPolarizingBeamSplitter, eta=jnp.pi / 4)
            fo_phase_shifter = Operation(FockOperationType.PhaseShift, phi=self.phase_shift)
            fo_external = Operation(FockOperationType.PhaseShift, phi=self.external_phase_shift)

            # implement a MZI with beam splitter, phase shifter, beam splitter
            # apply the beam splitter
            ce.apply_operation(fo_beam_splitter, env0.fock, env1.fock)

            # apply the phase shifter
            ce.apply_operation(fo_phase_shifter, env1.fock)

            # apply the beam splitter twice
            ce.apply_operation(fo_beam_splitter, env0.fock, env1.fock)

            # apply the external phase shifter
            ce.apply_operation(fo_external, env1.fock)

            # create new quantum signals
            qout0_signal = GenericQuantumSignal()
            qout1_signal = GenericQuantumSignal()

            # set the modified envelope to the content of output signal
            qout0_signal.set_contents(env0)
            qout1_signal.set_contents(env1)

            # return to the result
            result = [("qout0", qout0_signal, time), ("qout1", qout1_signal, time)]
            self.log_message(f"Unit Cell returning {kwargs['signals']}" )
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
