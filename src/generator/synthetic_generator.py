"""
Smart OBD-II Electrical Diagnostic System
Module: Synthetic Data Generator (Time-Series & Telemetry Simulation)

Simulates real-world automotive electrical telemetry via OBD-II (ELM327 / ESP32)
covering 4 core operational and failure modes:
1. Normal State (Healthy Battery & Alternator)
2. Battery Degradation / Weak Cranking
3. Alternator Failure (Undercharging, Overcharging, Diode AC Ripple)
4. Parasitic Drain & High Ground/Wiring Resistance
"""

import math
import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class SimulationPattern(IntEnum):
    NORMAL = 0
    BATTERY_DEGRADATION = 1
    ALTERNATOR_ISSUE = 2
    PARASITIC_DRAIN_OR_HIGH_RESISTANCE = 3


class EngineState(IntEnum):
    OFF = 0
    CRANKING = 1
    IDLE = 2
    DRIVING = 3


PATTERN_NAMES = {
    SimulationPattern.NORMAL: {
        "en": "Normal Healthy State",
        "ar": "الحالة الطبيعية السليمة",
    },
    SimulationPattern.BATTERY_DEGRADATION: {
        "en": "Battery Degradation / Weak Cranking",
        "ar": "تدهور البطارية / ضعف بدء التشغيل",
    },
    SimulationPattern.ALTERNATOR_ISSUE: {
        "en": "Alternator Fault (Under/Overcharge or Ripple)",
        "ar": "خلل في شحن الدينامو (شحن ضعيف/زائد أو تذبذب)",
    },
    SimulationPattern.PARASITIC_DRAIN_OR_HIGH_RESISTANCE: {
        "en": "Parasitic Drain / High Wiring Resistance",
        "ar": "تسريب تيار / مقاومة تأريض وأسلاك مرتفعة",
    },
}


@dataclass
class VehicleElectricalConfig:
    """Electrical characteristics of the simulated vehicle."""
    # Battery parameters
    nominal_voltage: float = 12.6         # Open-circuit healthy voltage (V)
    internal_resistance: float = 0.008    # Battery internal resistance (Ohms: 0.005 - 0.010 healthy)
    battery_capacity_ah: float = 60.0     # Nominal capacity (Ah)
    
    # Alternator parameters
    regulator_setpoint: float = 14.2      # Standard target voltage (V)
    alternator_max_amps: float = 110.0    # Alternator rating (Amps)
    alternator_health: float = 1.0        # 1.0 = 100% efficient, 0.0 = completely dead
    
    # Wiring & Grounds
    ground_resistance: float = 0.005      # Resistance between ground & chassis (Ohms)
    parasitic_drain_ma: float = 25.0      # Key-off parasitic draw (normal < 50mA)
    
    # Noise & Sensor Tolerances (ELM327 resolution ~0.1V, ADC precision)
    sensor_noise_std: float = 0.03


class OBDSyntheticDataGenerator:
    """
    Generates realistic time-series telemetry representing vehicle drive cycles.
    """

    def __init__(self, sampling_rate_hz: float = 5.0, random_seed: Optional[int] = 42):
        """
        :param sampling_rate_hz: Samples per second (OBD-II ELM327 typically yields 2-10 Hz)
        :param random_seed: Seed for reproducibility
        """
        self.sampling_rate_hz = sampling_rate_hz
        self.dt = 1.0 / sampling_rate_hz
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

    def generate_single_session(
        self,
        session_id: str,
        pattern: SimulationPattern,
        total_duration_sec: float = 60.0,
        sub_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generates a continuous drive session simulation containing:
        - Phase 1: Key ON / Engine OFF (Resting)
        - Phase 2: Engine Cranking (Starter active)
        - Phase 3: Engine Idle
        - Phase 4: Dynamic Driving / RPM & Speed variation
        - Phase 5: High Electrical Load (AC, Lights, Defogger on)
        - Phase 6: Return to Idle / Engine Shutdown
        """
        total_steps = int(total_duration_sec * self.sampling_rate_hz)
        
        # Determine timeline segment boundaries (in seconds)
        t_key_on = 0.0
        t_crank_start = random.uniform(3.0, 5.0)
        
        # Cranking duration: healthy ~0.8-1.2s, weak battery ~2.0-3.5s
        if pattern == SimulationPattern.BATTERY_DEGRADATION:
            crank_duration = random.uniform(2.0, 3.5)
        else:
            crank_duration = random.uniform(0.7, 1.3)
            
        t_crank_end = t_crank_start + crank_duration
        t_idle_end = t_crank_end + random.uniform(8.0, 15.0)
        t_driving_end = t_idle_end + random.uniform(15.0, 25.0)
        t_load_end = t_driving_end + random.uniform(12.0, 18.0)
        # Remaining time is shutdown / post-run
        
        # Setup failure parameters based on pattern & subtype
        config = self._configure_pattern_physics(pattern, sub_type)
        
        # State tracking variables
        records: List[Dict] = []
        current_soc = random.uniform(0.85, 0.98) if pattern != SimulationPattern.BATTERY_DEGRADATION else random.uniform(0.35, 0.65)
        current_temp_c = random.uniform(20.0, 35.0)
        engine_rpm = 0.0
        vehicle_speed_kmh = 0.0
        engine_load_pct = 0.0
        electrical_load_pct = 0.0
        
        # Dynamic memory for smoothing
        prev_voltage = config.nominal_voltage
        
        for step in range(total_steps):
            t = step * self.dt
            
            # --- 1. Determine Engine & Driving State ---
            if t < t_crank_start:
                # Key ON / Engine OFF
                state = EngineState.OFF
                engine_rpm = 0.0
                vehicle_speed_kmh = 0.0
                engine_load_pct = 0.0
                electrical_load_pct = random.uniform(5.0, 15.0)  # Dash lights, radio
                
            elif t < t_crank_end:
                # Engine Cranking
                state = EngineState.CRANKING
                engine_rpm = random.uniform(180.0, 280.0) if pattern != SimulationPattern.BATTERY_DEGRADATION else random.uniform(110.0, 190.0)
                vehicle_speed_kmh = 0.0
                engine_load_pct = 100.0
                electrical_load_pct = 10.0
                
            elif t < t_idle_end:
                # Post-start Idle
                state = EngineState.IDLE
                engine_rpm = 750.0 + 50.0 * math.sin(t * 0.5) + np.random.normal(0, 15)
                vehicle_speed_kmh = 0.0
                engine_load_pct = random.uniform(18.0, 25.0)
                electrical_load_pct = random.uniform(10.0, 20.0)
                current_temp_c += 0.05 * self.dt
                
            elif t < t_driving_end:
                # Dynamic Driving / Accelerations
                state = EngineState.DRIVING
                # Smooth sinusoidal + noise RPM profile
                drive_phase = (t - t_idle_end)
                engine_rpm = 1500.0 + 800.0 * math.sin(drive_phase * 0.3) + np.random.normal(0, 30)
                engine_rpm = max(750.0, engine_rpm)
                vehicle_speed_kmh = max(0.0, (engine_rpm - 750.0) * 0.035 + 20.0 * math.sin(drive_phase * 0.2))
                engine_load_pct = min(95.0, 25.0 + (engine_rpm / 3500.0) * 50.0 + np.random.normal(0, 5))
                electrical_load_pct = random.uniform(15.0, 30.0)
                current_temp_c += 0.1 * self.dt
                
            elif t < t_load_end:
                # Heavy Electrical Accessories Activated (A/C Max + High Beams + Rear Defrost)
                state = EngineState.DRIVING if (step % 2 == 0) else EngineState.IDLE
                engine_rpm = 1200.0 + 400.0 * math.cos((t - t_driving_end) * 0.4)
                vehicle_speed_kmh = max(0.0, (engine_rpm - 750.0) * 0.02)
                engine_load_pct = random.uniform(40.0, 65.0)
                electrical_load_pct = random.uniform(80.0, 98.0)  # Max accessory load
                current_temp_c += 0.08 * self.dt
                
            else:
                # Engine Cooldown / Shutdown
                state = EngineState.OFF
                engine_rpm = 0.0
                vehicle_speed_kmh = 0.0
                engine_load_pct = 0.0
                electrical_load_pct = 5.0
            
            # --- 2. Physics-Based Voltage Calculation ---
            voltage, current_amps, ripple_v = self._compute_instantaneous_voltage(
                state=state,
                rpm=engine_rpm,
                elec_load_pct=electrical_load_pct,
                t=t,
                config=config,
                pattern=pattern,
                sub_type=sub_type,
            )
            
            # Add sensor quantization and measurement noise
            measured_voltage = voltage + np.random.normal(0, config.sensor_noise_std)
            # Smooth physical inertial response
            measured_voltage = 0.85 * measured_voltage + 0.15 * prev_voltage
            prev_voltage = measured_voltage
            
            records.append({
                "session_id": session_id,
                "time_sec": round(t, 3),
                "engine_state": int(state),
                "rpm": round(float(engine_rpm), 1),
                "vehicle_speed_kmh": round(float(vehicle_speed_kmh), 1),
                "engine_load_pct": round(float(engine_load_pct), 1),
                "coolant_temp_c": round(float(current_temp_c), 1),
                "electrical_load_pct": round(float(electrical_load_pct), 1),
                "battery_voltage_v": round(float(measured_voltage), 3),
                "current_draw_a": round(float(current_amps), 2),
                "voltage_ripple_v": round(float(ripple_v), 3),
                "label": int(pattern),
                "label_name_en": PATTERN_NAMES[pattern]["en"],
                "label_name_ar": PATTERN_NAMES[pattern]["ar"],
                "anomaly_label": 0 if pattern == SimulationPattern.NORMAL else 1,
            })
            
        df = pd.DataFrame(records)
        return df

    def _configure_pattern_physics(
        self,
        pattern: SimulationPattern,
        sub_type: Optional[str],
    ) -> VehicleElectricalConfig:
        """Configures physical electrical parameters for specific fault scenarios."""
        cfg = VehicleElectricalConfig()
        
        if pattern == SimulationPattern.NORMAL:
            cfg.nominal_voltage = random.uniform(12.45, 12.75)
            cfg.internal_resistance = random.uniform(0.005, 0.009)
            cfg.regulator_setpoint = random.uniform(14.0, 14.35)
            cfg.alternator_health = 1.0
            cfg.ground_resistance = random.uniform(0.002, 0.008)
            cfg.parasitic_drain_ma = random.uniform(15.0, 35.0)
            
        elif pattern == SimulationPattern.BATTERY_DEGRADATION:
            # Aged battery: low resting voltage and high internal resistance
            cfg.nominal_voltage = random.uniform(11.5, 12.15)
            cfg.internal_resistance = random.uniform(0.028, 0.055)  # 3x-6x higher!
            cfg.regulator_setpoint = random.uniform(13.9, 14.3)
            cfg.alternator_health = 1.0
            cfg.ground_resistance = random.uniform(0.005, 0.010)
            cfg.parasitic_drain_ma = random.uniform(20.0, 45.0)
            
        elif pattern == SimulationPattern.ALTERNATOR_ISSUE:
            # Subtypes: undercharging, overcharging, or diode AC ripple
            chosen_sub = sub_type or random.choice(["undercharge", "overcharge", "ripple"])
            if chosen_sub == "undercharge":
                cfg.nominal_voltage = random.uniform(12.2, 12.5)
                cfg.internal_resistance = random.uniform(0.008, 0.012)
                cfg.regulator_setpoint = random.uniform(11.8, 13.1)  # Weak charging!
                cfg.alternator_health = random.uniform(0.2, 0.6)
            elif chosen_sub == "overcharge":
                cfg.nominal_voltage = random.uniform(12.5, 12.7)
                cfg.internal_resistance = random.uniform(0.006, 0.010)
                cfg.regulator_setpoint = random.uniform(15.2, 16.5)  # Dangerous overvoltage!
                cfg.alternator_health = 1.2
            else:  # ripple
                cfg.nominal_voltage = random.uniform(12.4, 12.6)
                cfg.internal_resistance = random.uniform(0.008, 0.012)
                cfg.regulator_setpoint = random.uniform(13.8, 14.2)
                cfg.alternator_health = 0.85
                
        elif pattern == SimulationPattern.PARASITIC_DRAIN_OR_HIGH_RESISTANCE:
            chosen_sub = sub_type or random.choice(["parasitic_drain", "high_ground_resistance"])
            if chosen_sub == "parasitic_drain":
                cfg.nominal_voltage = random.uniform(12.1, 12.4)
                cfg.internal_resistance = random.uniform(0.008, 0.012)
                cfg.regulator_setpoint = random.uniform(13.9, 14.3)
                cfg.parasitic_drain_ma = random.uniform(400.0, 1800.0)  # Huge drain (0.4A - 1.8A)
            else:  # high_ground_resistance
                cfg.nominal_voltage = random.uniform(12.5, 12.7)
                cfg.internal_resistance = random.uniform(0.008, 0.012)
                cfg.regulator_setpoint = random.uniform(14.0, 14.4)
                cfg.ground_resistance = random.uniform(0.08, 0.22)  # High contact/chassis resistance
                
        return cfg

    def _compute_instantaneous_voltage(
        self,
        state: EngineState,
        rpm: float,
        elec_load_pct: float,
        t: float,
        config: VehicleElectricalConfig,
        pattern: SimulationPattern,
        sub_type: Optional[str],
    ) -> Tuple[float, float, float]:
        """Calculates instantaneous terminal voltage, current, and ripple."""
        
        ripple_v = random.uniform(0.01, 0.03)  # baseline ripple
        
        # Base accessory load current (e.g. 0 to 60 Amps)
        accessory_amps = (elec_load_pct / 100.0) * 55.0
        
        if state == EngineState.OFF:
            # Parasitic drain impact
            drain_amps = (config.parasitic_drain_ma / 1000.0)
            total_current = drain_amps + accessory_amps * 0.1  # small idle accessories
            # Continuous discharge drop
            decay = 0.0
            if pattern == SimulationPattern.PARASITIC_DRAIN_OR_HIGH_RESISTANCE:
                # Fast decaying resting voltage under parasitic drain
                decay = 0.008 * t
            
            # V_terminal = V_oc - I * (R_int + R_ground)
            voltage = config.nominal_voltage - decay - (total_current * config.internal_resistance)
            return voltage, total_current, ripple_v
            
        elif state == EngineState.CRANKING:
            # Cranking starter inrush current: 200A - 450A
            if pattern == SimulationPattern.BATTERY_DEGRADATION:
                cranking_amps = random.uniform(280.0, 420.0)
            else:
                cranking_amps = random.uniform(220.0, 320.0)
                
            total_current = cranking_amps + accessory_amps
            
            # Significant voltage drop across internal resistance: V = V_oc - I * R_int
            voltage_drop = total_current * (config.internal_resistance + config.ground_resistance)
            voltage = config.nominal_voltage - voltage_drop
            
            # Minimum physics bounds
            if pattern == SimulationPattern.BATTERY_DEGRADATION:
                voltage = max(6.8, min(voltage, 9.4))  # Fails < 9.6V threshold
            else:
                voltage = max(9.8, min(voltage, 11.4))  # Healthy > 9.6V threshold
                
            return voltage, total_current, ripple_v * 1.5

        else:
            # Engine Running (IDLE or DRIVING)
            # Alternator is rotating and generating power
            # Alternator output capacity scales with RPM
            rpm_ratio = min(1.0, max(0.4, rpm / 2000.0))
            available_alt_amps = config.alternator_max_amps * rpm_ratio * config.alternator_health
            
            net_charging_demand = accessory_amps + 15.0  # Vehicle ignition & battery charge demand
            
            if pattern == SimulationPattern.ALTERNATOR_ISSUE:
                if config.regulator_setpoint > 15.0:
                    # Overcharging scenario
                    target_v = config.regulator_setpoint + random.uniform(-0.1, 0.2)
                    net_current = -25.0  # High forced charging current
                elif config.regulator_setpoint < 13.2:
                    # Undercharging scenario
                    # Alternator cannot sustain load, battery helps discharge
                    target_v = config.regulator_setpoint - (accessory_amps * 0.015)
                    net_current = accessory_amps - available_alt_amps
                else:
                    # Diode ripple scenario (AC noise on DC line)
                    target_v = config.regulator_setpoint
                    ripple_v = random.uniform(0.35, 0.85)  # High ripple noise!
                    target_v += ripple_v * math.sin(2 * math.pi * 15 * t)
                    net_current = -10.0
                voltage = target_v
            
            elif pattern == SimulationPattern.PARASITIC_DRAIN_OR_HIGH_RESISTANCE and config.ground_resistance > 0.05:
                # High contact resistance causes huge load-dependent voltage drop
                voltage_drop_ground = accessory_amps * config.ground_resistance
                target_v = config.regulator_setpoint - voltage_drop_ground
                voltage = target_v
                net_current = -12.0
                
            else:
                # Healthy Charging State
                # Slight temporary load dip that recovers
                target_v = config.regulator_setpoint - (accessory_amps * 0.003)
                voltage = target_v
                net_current = -(available_alt_amps - accessory_amps) * 0.2  # Charging battery
                
            return voltage, net_current, ripple_v

    def generate_full_dataset(
        self,
        sessions_per_pattern: int = 50,
        session_duration_sec: float = 60.0,
        shuffle: bool = True,
    ) -> pd.DataFrame:
        """
        Generates a balanced, comprehensive dataset covering all 4 diagnostic classes.
        """
        all_dfs = []
        session_counter = 1
        
        for pattern in SimulationPattern:
            subtypes = [None]
            if pattern == SimulationPattern.ALTERNATOR_ISSUE:
                subtypes = ["undercharge", "overcharge", "ripple"]
            elif pattern == SimulationPattern.PARASITIC_DRAIN_OR_HIGH_RESISTANCE:
                subtypes = ["parasitic_drain", "high_ground_resistance"]
                
            for i in range(sessions_per_pattern):
                sub = subtypes[i % len(subtypes)]
                session_id = f"SESSION_{session_counter:04d}_PAT_{pattern.value}_{pattern.name.lower()}"
                session_df = self.generate_single_session(
                    session_id=session_id,
                    pattern=pattern,
                    total_duration_sec=session_duration_sec,
                    sub_type=sub,
                )
                all_dfs.append(session_df)
                session_counter += 1
                
        full_df = pd.concat(all_dfs, ignore_index=True)
        if shuffle:
            # Keep sessions together or row shuffle based on need
            # Here we preserve time-series ordering within sessions
            pass
        return full_df
