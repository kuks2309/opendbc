from opendbc.car import DT_CTRL
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.tesla.values import CANBUS, CarControllerParams, TeslaFlags


def get_steer_ctrl_type(flags: int, ctrl_type: int) -> int:
  # Returns the flipped signal value for DAS_steeringControlType on FSD 14
  if flags & TeslaFlags.FSD_14:
    return {1: 2, 2: 1}.get(ctrl_type, ctrl_type)
  else:
    return ctrl_type


class TeslaCAN:
  def __init__(self, CP, packer):
    self.CP = CP
    self.packer = packer
    self.jerk = 0.0

  def create_steering_control(self, angle, enabled):
    # On FSD 14+, ANGLE_CONTROL behavior changed to allow user winddown while actuating.
    # with openpilot, after overriding w/ ANGLE_CONTROL the wheel snaps back to the original angle abruptly
    # so we now use LANE_KEEP_ASSIST to match stock FSD.
    # see carstate.py for more details
    values = {
      "DAS_steeringAngleRequest": -angle,
      "DAS_steeringHapticRequest": 0,
      "DAS_steeringControlType": get_steer_ctrl_type(self.CP.flags, 1 if enabled else 0),
    }

    return self.packer.make_can_msg("DAS_steeringControl", CANBUS.party, values)

  def create_longitudinal_command(self, acc_state, accel, counter, v_ego, active, cruise_override):
    set_speed = min(max(v_ego + accel, 0) * CV.MS_TO_KPH, 400)

    # ramping max jerk fixes jerkiness after gas override when above max speed
    self.jerk = 0 if cruise_override else (self.jerk + CarControllerParams.JERK_RATE_UP * DT_CTRL * 4)

    values = {
      "DAS_setSpeed": set_speed,
      "DAS_accState": acc_state,
      "DAS_aebEvent": 0,
      "DAS_jerkMin": CarControllerParams.JERK_LIMIT_MIN,
      "DAS_jerkMax": min(self.jerk, CarControllerParams.JERK_LIMIT_MAX), # ramping max jerk is enough for some reason
      "DAS_accelMin": accel,
      "DAS_accelMax": max(accel, 0),
      "DAS_controlCounter": counter,
    }
    return self.packer.make_can_msg("DAS_control", CANBUS.party, values)

  def create_longitudinal_passthrough(self, das_control, counter):
    # Delegate longitudinal to Tesla: relay Tesla's own DAS_control command UNCHANGED (no blend),
    # so the car does exactly what Tesla's TACC intends. accel bounds are clipped to openpilot's
    # authority so panda safety accepts the message.
    lo, hi = CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX
    values = {
      "DAS_setSpeed": das_control["DAS_setSpeed"],
      "DAS_accState": das_control["DAS_accState"],
      "DAS_aebEvent": das_control["DAS_aebEvent"],
      "DAS_jerkMin": das_control["DAS_jerkMin"],
      "DAS_jerkMax": das_control["DAS_jerkMax"],
      "DAS_accelMin": max(lo, min(hi, das_control["DAS_accelMin"])),
      "DAS_accelMax": max(lo, min(hi, das_control["DAS_accelMax"])),
      "DAS_controlCounter": counter,
    }
    return self.packer.make_can_msg("DAS_control", CANBUS.party, values)

  def create_longitudinal_blended(self, accel, v_ego, das_control, blend, counter):
    # Smoothly transition between openpilot's command (blend=0) and Tesla's DAS_control (blend=1)
    # to reduce jerk when there is time (TTC long). Used only during transitions; steady state is
    # pure openpilot or pure Tesla passthrough.
    lo, hi = CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX
    b = min(max(blend, 0.0), 1.0)
    op_set = min(max(v_ego + accel, 0) * CV.MS_TO_KPH, 400)
    self.jerk = self.jerk + CarControllerParams.JERK_RATE_UP * DT_CTRL * 4
    op_jmax = min(self.jerk, CarControllerParams.JERK_LIMIT_MAX)
    values = {
      "DAS_setSpeed": (1 - b) * op_set + b * das_control["DAS_setSpeed"],
      "DAS_accState": 4,
      "DAS_aebEvent": das_control["DAS_aebEvent"] if b > 0.5 else 0,
      "DAS_jerkMin": (1 - b) * CarControllerParams.JERK_LIMIT_MIN + b * das_control["DAS_jerkMin"],
      "DAS_jerkMax": (1 - b) * op_jmax + b * das_control["DAS_jerkMax"],
      "DAS_accelMin": max(lo, min(hi, (1 - b) * accel + b * das_control["DAS_accelMin"])),
      "DAS_accelMax": max(lo, min(hi, (1 - b) * max(accel, 0) + b * das_control["DAS_accelMax"])),
      "DAS_controlCounter": counter,
    }
    return self.packer.make_can_msg("DAS_control", CANBUS.party, values)

  def create_steering_allowed(self):
    values = {
      "APS_eacAllow": 1,
    }

    return self.packer.make_can_msg("APS_eacMonitor", CANBUS.party, values)


def tesla_checksum(address: int, sig, d: bytearray) -> int:
  checksum = (address & 0xFF) + ((address >> 8) & 0xFF)
  checksum_byte = sig.start_bit // 8
  for i in range(len(d)):
    if i != checksum_byte:
      checksum += d[i]
  return checksum & 0xFF
