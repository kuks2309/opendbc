import numpy as np
from collections import deque
from opendbc.can import CANPacker
from opendbc.car import Bus
from opendbc.car.lateral import apply_steer_angle_limits_vm
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.tesla.teslacan import TeslaCAN
from opendbc.car.tesla.values import CarControllerParams
from opendbc.car.vehicle_model import VehicleModel
from opendbc.sunnypilot.car.tesla.coop_steering import CoopSteeringCarController
from opendbc.sunnypilot.car.tesla.values import TeslaFlagsSP

# Tesla ACC longitudinal fusion (experimental A/B) tuning
DAS_ACC_ON = 4                        # DAS_accState enum value for ACC_ON
LONG_FUSION_LEAD_HOLD = 25            # longitudinal ticks (~1s @25Hz) to keep delegating after lead flickers off
# Emergency override: Tesla's accelMax is its MAX allowed accel; when it drops strongly negative Tesla is
# FORCING a hard brake. Delegate to Tesla then even if openpilot missed the lead. (Needs field calibration
# against a real Tesla hard brake; normal driving keeps accelMax > 0.)
TESLA_EMERGENCY_ACCELMAX = -1.5       # m/s^2
LONG_FUSION_BLEND_STEP = 0.04         # per longitudinal tick (~1s ramp @25Hz) for smooth transition
LONG_FUSION_TTC_HARD = 4.0            # s; time-to-collision below this = collision expected -> hard switch (no ramp)
# Curve assist (#4): Juniper firmware broadcasts no discrete curve-assist flag (ACC_report/csaState are
# frozen; full bus bit-correlation scan 2026-07-18 found none). The curve profile lives in DAS_setSpeed
# itself: Tesla drops its dynamic target below current speed several seconds before the corner.
# Gate arms only on a RECENT RAPID setSpeed drop -- highway logs show a static 3-13 kph deficit
# (stale stock set speed / speedo offset) that a plain threshold would false-trigger on for minutes.
LONG_FUSION_SLOW_MARGIN = 3.0 / 3.6   # m/s; setSpeed this far below vEgo = Tesla wants to slow (curve/limit)
LONG_FUSION_SLOW_STAY = 1.0 / 3.6     # m/s; latched gate releases when setSpeed recovers to vEgo - this
LONG_FUSION_SLOW_DROP_KPH = 5.0       # kph; setSpeed must have fallen this much within the window to arm
LONG_FUSION_SLOW_DROP_WIN = 50        # longitudinal ticks (~2s @25Hz) drop-detection window
LONG_FUSION_SLOW_VMIN = 11.0          # m/s (~40 km/h); below this keep openpilot (city stop-and-go noise)
LONG_FUSION_SLOW_HOLD = 25            # longitudinal ticks (~1s @25Hz) to ride out setSpeed jitter


def get_safety_CP():
  # We use the TESLA_MODEL_Y platform for lateral limiting to match safety
  # A Model 3 at 40 m/s using the Model Y limits sees a <0.3% difference in max angle (from curvature factor)
  from opendbc.car.tesla.interface import CarInterface
  return CarInterface.get_non_essential_params("TESLA_MODEL_Y")


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP, CP_SP):
    CarControllerBase.__init__(self, dbc_names, CP, CP_SP)
    self.coop_steer = CoopSteeringCarController()
    self.apply_angle_last = 0
    self.lead_hold_frames = 0   # hysteresis for Tesla long fusion lead gating
    self.slow_hold_frames = 0   # hysteresis for Tesla curve-assist (setSpeed deficit) gating
    self.slow_latched = False   # curve-assist gate latch (armed by rapid setSpeed drop)
    self.set_hist = deque(maxlen=LONG_FUSION_SLOW_DROP_WIN)  # recent DAS_setSpeed (kph) for drop detection
    self.deleg_blend = 0.0      # 0=openpilot, 1=Tesla; ramps for smooth transition (snaps on collision)
    self.packer = CANPacker(dbc_names[Bus.party])
    self.tesla_can = TeslaCAN(CP, self.packer)

    # Vehicle model used for lateral limiting
    self.VM = VehicleModel(get_safety_CP())

  def update(self, CC, CC_SP, CS, now_nanos):
    actuators = CC.actuators
    can_sends = []

    # Wait until the override condition clears before steering
    # Canceling is done on rising edge of CS.out.steeringDisengage and is handled generically with CC.cruiseControl.cancel
    lat_active = CC.latActive and not CS.out.steeringDisengage

    if self.frame % 2 == 0:
      # Angular rate limit based on speed
      self.apply_angle_last = apply_steer_angle_limits_vm(actuators.steeringAngleDeg, self.apply_angle_last, CS.out.vEgoRaw, CS.out.steeringAngleDeg,
                                                          lat_active, CarControllerParams, self.VM)

      can_sends.append(self.tesla_can.create_steering_control(*self.coop_steer.update(self.apply_angle_last, lat_active, self.CP_SP, CS, self.VM)))

    if self.frame % 10 == 0:
      can_sends.append(self.tesla_can.create_steering_allowed())

    # Longitudinal control
    if self.CP.openpilotLongitudinalControl:
      if self.frame % 4 == 0:
        cntr = (self.frame // 4) % 8

        # Tesla longitudinal delegation (A/B experimental option, toggle OFF => byte-identical stock behavior).
        # When openpilot sees a lead (hudControl.leadVisible = the on-screen lead chevron), hand longitudinal
        # to Tesla by relaying Tesla's own DAS_control command UNCHANGED (a clean switch/pass-through, NOT a
        # blend). Otherwise openpilot controls. A short hold rides out lead-detection flicker.
        delegate = False
        collision_soon = False
        if self.CP_SP.flags & TeslaFlagsSP.TESLA_LONG_FUSION.value:
          if CC.hudControl.leadVisible:
            self.lead_hold_frames = LONG_FUSION_LEAD_HOLD
          elif self.lead_hold_frames > 0:
            self.lead_hold_frames -= 1
          lead_present = CC.hudControl.leadVisible or self.lead_hold_frames > 0
          tesla_ok = CS.das_control is not None and CS.das_control["DAS_accState"] == DAS_ACC_ON
          # #5 Emergency override: Tesla FORCING a hard brake (accelMax < threshold) delegates even if
          # openpilot did not see the lead -- catches leads openpilot's vision misses (7/5 incident).
          tesla_emergency = tesla_ok and CS.das_control["DAS_accelMax"] < TESLA_EMERGENCY_ACCELMAX
          # #4 Curve assist: Tesla embeds the corner slowdown in DAS_setSpeed (drops below current
          # speed ahead of the corner). Delegate so Tesla's own curve deceleration executes -- vision
          # alone sees the curve too late (user report 7/18). Armed only by a rapid recent drop,
          # latched while the deficit persists, released once setSpeed recovers near vEgo.
          if tesla_ok:
            das_set = CS.das_control["DAS_setSpeed"]
            self.set_hist.append(das_set)
            at_speed = CS.out.vEgo > LONG_FUSION_SLOW_VMIN
            recent_drop = (max(self.set_hist) - das_set) >= LONG_FUSION_SLOW_DROP_KPH
            deficit = at_speed and das_set / 3.6 < CS.out.vEgo - LONG_FUSION_SLOW_MARGIN
            staying = at_speed and das_set / 3.6 < CS.out.vEgo - LONG_FUSION_SLOW_STAY
            if deficit and recent_drop:
              self.slow_latched = True
            if not staying:
              self.slow_latched = False
          else:
            self.set_hist.clear()
            self.slow_latched = False
          if self.slow_latched:
            self.slow_hold_frames = LONG_FUSION_SLOW_HOLD
          elif self.slow_hold_frames > 0:
            self.slow_hold_frames -= 1
          tesla_slow = self.slow_latched or self.slow_hold_frames > 0
          # #3 Lane change (openpilot auto lane change sets these blinkers): openpilot keeps longitudinal,
          # but a Tesla emergency brake still overrides.
          lane_changing = CC.leftBlinker or CC.rightBlinker
          delegate = (CC.longActive and not CC.cruiseControl.cancel and tesla_ok and
                      (((lead_present or tesla_slow) and not lane_changing) or tesla_emergency))
          # Collision expected: short time-to-collision to the lead (dRel/closing) or Tesla emergency.
          lead = CC_SP.leadOne
          closing = -lead.vRel
          ttc = (lead.dRel / closing) if (lead.status and closing > 0.5) else 1e3
          collision_soon = tesla_emergency or ttc < LONG_FUSION_TTC_HARD

        # Ramp the blend toward the delegate target to reduce jerk; snap to full on collision (hard switch).
        if delegate and collision_soon:
          self.deleg_blend = 1.0
        elif delegate:
          self.deleg_blend = min(1.0, self.deleg_blend + LONG_FUSION_BLEND_STEP)
        else:
          self.deleg_blend = max(0.0, self.deleg_blend - LONG_FUSION_BLEND_STEP)

        use_tesla = self.deleg_blend > 0.001 and CS.das_control is not None
        accel = float(np.clip(actuators.accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
        if CC.cruiseControl.cancel:                                 # cancel outranks delegation
          # The cancel edge lives only ~0.4s. deleg_blend ramps down over more frames than that, so
          # while it is still > 0 we would stay in the pass-through/blend branches below -- neither of
          # which emits the cancel -- and Tesla's ACC stays engaged after openpilot drops out, which
          # faults the DI (cruise fault, restart required) on steering-override disengage. Snap out of
          # delegation and send the cancel on the same frame it is requested.
          self.deleg_blend = 0.0
          can_sends.append(self.tesla_can.create_longitudinal_command(13, accel, cntr, CS.out.vEgo, CC.longActive, CS.cruise_override))
        elif use_tesla and self.deleg_blend >= 0.999:               # #2 pure Tesla pass-through
          can_sends.append(self.tesla_can.create_longitudinal_passthrough(CS.das_control, cntr))
        elif use_tesla:                                             # transition: smooth blend
          can_sends.append(self.tesla_can.create_longitudinal_blended(accel, CS.out.vEgo, CS.das_control, self.deleg_blend, cntr))
        else:                                                       # #1/#4 openpilot
          can_sends.append(self.tesla_can.create_longitudinal_command(4, accel, cntr, CS.out.vEgo, CC.longActive, CS.cruise_override))  # 4=ACC_ON

    else:
      # Increment counter so cancel is prioritized even without openpilot longitudinal
      if CC.cruiseControl.cancel:
        cntr = (CS.das_control["DAS_controlCounter"] + 1) % 8
        can_sends.append(self.tesla_can.create_longitudinal_command(13, 0, cntr, CS.out.vEgo, False, True))

    # TODO: HUD control
    new_actuators = actuators.as_builder()
    new_actuators.steeringAngleDeg = self.apply_angle_last
    new_actuators.accel = self.coop_steer.coop_apply_angle_sat_last # debug
    new_actuators.curvature = float(self.coop_steer.debug_angle_desired_limited) # debug
    new_actuators.torque = float(self.coop_steer.angle_override) # debug

    self.frame += 1
    return new_actuators, can_sends
