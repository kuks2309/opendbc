"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from enum import IntFlag


class TeslaFlagsSP(IntFlag):
  HAS_VEHICLE_BUS = 1  # 3-finger infotainment press signal is present on the VEHICLE bus with the deprecated Tesla harness installed
  COOP_STEERING = 2  # Coop steering
  TESLA_LONG_FUSION = 4  # Floor openpilot's commanded accel with Tesla's own TACC decel command (A/B experimental option)
  TESLA_CURVE_SLOW = 8  # Delegate to Tesla when its DAS_setSpeed drops for curve assist (requires TESLA_LONG_FUSION)


class TeslaSafetyFlagsSP:
  HAS_VEHICLE_BUS = 1
