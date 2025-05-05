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
import copy


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
    def des(self, time, *args, **kwargs):
        try:
            def apply_unit_cell(env0, env1):

                # env0 = qin0_signal.contents if qin0_signal else Envelope()
                # env1 = qin1_signal.contents if qin1_signal else Envelope()

                try:
                    ce = CompositeEnvelope(env0, env1)
                    ce.combine(env0.fock, env1.fock)
                    self.log_message(f"composite envelope: {len(ce.states)}")
                    # define some operation
                    fo_beam_splitter = Operation(CompositeOperationType.NonPolarizingBeamSplitter, eta=jnp.pi / 4)
                    fo_phase_shifter = Operation(FockOperationType.PhaseShift, phi=self.phase_shift)
                    fo_external = Operation(FockOperationType.PhaseShift, phi=self.external_phase_shift)
                    # apply some operation
                    ce.apply_operation(fo_beam_splitter, env0.fock, env1.fock)
                    ce.apply_operation(fo_phase_shifter, env1.fock)
                    ce.apply_operation(fo_beam_splitter, env0.fock, env1.fock)
                    ce.apply_operation(fo_external, env1.fock)

                    # qout0 = GenericQuantumSignal()
                    # qout1 = GenericQuantumSignal()
                    # qout0.set_contents(env0)
                    # qout1.set_contents(env1)
                    return env0, env1
                except Exception as e:
                    self.log_message(f"apply unit cell error: {traceback.format_exc()}")

            if time is None:
                time = 0.0
            time = float(time)

            signals = kwargs.get("signals", {})
            # phase shift parameter
            if "phase_shift" in kwargs.get("signals"):
                self.phase_shift = kwargs["signals"]["phase_shift"].contents
            if "external_phase_shift" in kwargs.get("signals"):
                self.external_phase_shift = kwargs["signals"]["external_phase_shift"].contents

            # get the envelopes
            qin0_signal = signals.get("qin0", None)
            qin1_signal = signals.get("qin1", None)
            qin2_signal = signals.get("qin2", None)
            qin3_signal = signals.get("qin3", None)
            # phase_shift = signals.get("phase_shift", None)
            # external_phase_shift = signals.get("external_phase_shift", None)

            # envelope
            if qin0_signal is None:
                env0 = Envelope()
            else:
                env0 = signals["qin0"].contents
            if qin1_signal is None:
                env1 = Envelope()
            else:
                env1 = signals["qin1"].contents
            if qin2_signal is None:
                env2 = Envelope()
            else:
                env2 = signals["qin2"].contents
            if qin3_signal is None:
                env3 = Envelope()
            else:
                env3 = signals["qin3"].contents

            if (qin0_signal is None) and (qin1_signal is None) and (qin2_signal is None) and (qin3_signal is None):
                print("MZI has no input")
                self.log_message("MZI no input here")

                return

            print(f"qin0_signal: {env0}, qin1_signal: {env1}, qin2_signal: {env2}, qin3_signal: {env3}")
            self.log_message(f"env0: {env0}, env1: {env1}, env2: {env2}, env3: {env3}")

            # apply unit cell
            u1_out0, u1_out1 = apply_unit_cell(env0, env1)
            u2_out0, u2_out1 = apply_unit_cell(env2, env3)
            u3_out0, u3_out1 = apply_unit_cell(u1_out1, u2_out0)
            u4_out0, u4_out1 = apply_unit_cell(u1_out0, u3_out0)
            u5_out0, u5_out1 = apply_unit_cell(u2_out1, u3_out1)
            u6_out0, u6_out1 = apply_unit_cell(u4_out1, u5_out0)

            # processor输出
            qout0_signal = GenericQuantumSignal()
            qout1_signal = GenericQuantumSignal()
            qout2_signal = GenericQuantumSignal()
            qout3_signal = GenericQuantumSignal()

            qout0_signal.set_contents(u4_out0)
            qout1_signal.set_contents(u6_out0)
            qout2_signal.set_contents(u6_out1)
            qout3_signal.set_contents(u5_out1)

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