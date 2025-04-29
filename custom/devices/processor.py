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


class Processor(GenericDevice):
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
        
        "qin2": Port(
            label="qin2",
            direction="input",
            signal=None,
            signal_type=qureed.signals.generic_quantum_signal.GenericQuantumSignal,
            device=None),
        
        "qin3": Port(
            label="qin3",
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
        
        "qout2": Port(
            label="qout2",
            direction="output",  # Changed from "input" to "output" for clarity
            signal=None,
            signal_type=qureed.signals.generic_quantum_signal.GenericQuantumSignal,
            device=None),
        
        "qout3": Port(
            label="qout3",
            direction="output",  # Changed from "input" to "output" for clarity
            signal=None,
            signal_type=qureed.signals.generic_quantum_signal.GenericQuantumSignal,
            device=None),
        
    }

    properties = {
	
    }
    

    gui_icon = HISTOGRAM

    gui_name = "Processor"
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
    def apply_unit_cell(self, qin0_signal, qin1_signal, phase_shift, external_phase_shift, time=None, *args, **kwargs):

        if time is None:
            time = 0.0
        time = float(time)

        #signals = kwargs.get("signals", {})
        # get the envelopes
        #qin0_signal = signals.get("qin0", None)
        #qin1_signal = signals.get("qin1", None)


        if qin0_signal is None:
            env0 = Envelope()
        else:
            #env0 = signals["qin0"].contents
            env0 = qin0_signal.contents

        if qin1_signal is None:
            env1 = Envelope()
        else:
            #env1 = signals["qin1"].contents
            env1 = qin1_signal.contents

        if (qin0_signal is None) and (qin1_signal is None):
            print("MZI has no input")
            self.log_message("MZI no input here")
            qout0 = GenericQuantumSignal()
            qout1 = GenericQuantumSignal()
            qout0.set_contents(Envelope())
            qout1.set_contents(Envelope())
            return qout1, qout0
            #return

        ce = CompositeEnvelope(env0, env1)
        ce.combine(env0.fock, env1.fock)

        fo_beam_splitter = Operation(CompositeOperationType.NonPolarizingBeamSplitter, eta=jnp.pi / 4)
        fo_phase_shifter = Operation(FockOperationType.PhaseShift, phi=self.phase_shift)
        fo_external = Operation(FockOperationType.PhaseShift, phi=self.external_phase_shift)

        ce.apply_operation(fo_beam_splitter, env0.fock, env1.fock)
        ce.apply_operation(fo_phase_shifter, env1.fock)
        ce.apply_operation(fo_beam_splitter, env0.fock, env1.fock)
        ce.apply_operation(fo_external, env1.fock)

        qout0_signal = GenericQuantumSignal()
        qout1_signal = GenericQuantumSignal()
        qout0_signal.set_contents(env0)
        qout1_signal.set_contents(env1)
        return qout0_signal, qout1_signal

    def des(self, time=None, *args, **kwargs):
        """
        Implement to use discrete event simulation
        """
        try:
            if time is None:
                time = 0.0
            time = float(time)

            signals = kwargs.get("signals", {})

            if "phase_shift" in kwargs.get("signals"):
                self.phase_shift = kwargs["signals"]["phase_shift"].contents

            if "external_phase_shift" in kwargs.get("signals"):
                self.external_phase_shift = kwargs["signals"]["external_phase_shift"].contents

            phase_shift = signals.get("phase_shift", None)
            external_phase_shift = signals.get("external_phase_shift", None)
            qin0_signal = signals.get("qin0", None)
            qin1_signal = signals.get("qin1", None)
            qin2_signal = signals.get("qin2", None)
            qin3_signal = signals.get("qin3", None)

############################### test ##########################################################
            if qin0_signal is None and qin1_signal is None:
                raise ValueError("input signals for unit1 is missing")
##################################################################################################

            unit1_qout0, unit1_qout1 = self.apply_unit_cell(qin0_signal, qin1_signal, self.phase_shift,
                                                            self.external_phase_shift)
            unit2_qout0, unit2_qout1 = self.apply_unit_cell(qin2_signal, qin3_signal, self.phase_shift,
                                                            self.external_phase_shift)
            unit3_qout0, unit3_qout1 = self.apply_unit_cell(unit1_qout1, unit2_qout0, phase_shift, external_phase_shift)
            unit4_qout0, unit4_qout1 = self.apply_unit_cell(unit1_qout0, unit3_qout0, phase_shift, external_phase_shift)
            unit5_qout0, unit5_qout1 = self.apply_unit_cell(unit3_qout1, unit2_qout1, phase_shift, external_phase_shift)

            # create new quantum signals
            qout0_signal = GenericQuantumSignal()
            qout1_signal = GenericQuantumSignal()
            qout2_signal = GenericQuantumSignal()
            qout3_signal = GenericQuantumSignal()

            # set the modified envelope to the content of output signal
            #qout0_signal.set_contents(unit4_qout0)
            qout0_signal.set_contents(unit4_qout0.contents)
            qout1_signal.set_contents(unit4_qout1.contents)
            qout2_signal.set_contents(unit5_qout0.contents)
            qout3_signal.set_contents(unit5_qout1.contents)

            result = [("qout0", qout0_signal, time), ("qout1", qout1_signal, time),
                      ("qout2", qout2_signal, time), ("qout3", qout3_signal, time)]
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