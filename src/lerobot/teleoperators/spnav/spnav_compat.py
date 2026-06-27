#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal Python 3 compatible libspnav wrapper for SpaceMouse teleoperation."""

from ctypes import POINTER, Structure, Union, c_int, c_uint, c_void_p, cdll, pointer

SPNAV_EVENT_ANY = 0
SPNAV_EVENT_MOTION = 1
SPNAV_EVENT_BUTTON = 2


class SpnavException(Exception):
    pass


class SpnavConnectionException(SpnavException):
    pass


class SpnavWaitException(SpnavException):
    pass


class SpnavEvent:
    pass


class SpnavMotionEvent(SpnavEvent):
    def __init__(self, translation, rotation, period: int):
        self.translation = translation
        self.rotation = rotation
        self.period = period


class SpnavButtonEvent(SpnavEvent):
    def __init__(self, bnum: int, press: int):
        self.bnum = bnum
        self.press = press


class spnav_event_motion(Structure):
    _fields_ = [
        ("type", c_int),
        ("x", c_int),
        ("y", c_int),
        ("z", c_int),
        ("rx", c_int),
        ("ry", c_int),
        ("rz", c_int),
        ("period", c_uint),
        ("data", c_void_p),
    ]


class spnav_event_button(Structure):
    _fields_ = [
        ("type", c_int),
        ("press", c_int),
        ("bnum", c_int),
    ]


class spnav_event(Union):
    _fields_ = [
        ("type", c_int),
        ("motion", spnav_event_motion),
        ("button", spnav_event_button),
    ]


libspnav = cdll.LoadLibrary("libspnav.so")
libspnav.spnav_open.restype = c_int
libspnav.spnav_fd.restype = c_int
libspnav.spnav_poll_event.argtypes = [POINTER(spnav_event)]
libspnav.spnav_poll_event.restype = c_int
libspnav.spnav_remove_events.argtypes = [c_int]
libspnav.spnav_remove_events.restype = c_int


def _convert_event(event: spnav_event) -> SpnavEvent | None:
    if event.type == SPNAV_EVENT_MOTION:
        return SpnavMotionEvent(
            [event.motion.x, event.motion.y, event.motion.z],
            [event.motion.rx, event.motion.ry, event.motion.rz],
            event.motion.period,
        )
    if event.type == SPNAV_EVENT_BUTTON:
        return SpnavButtonEvent(event.button.bnum, event.button.press)
    raise SpnavException(f"Invalid spnav event type: {event.type}")


def spnav_open() -> None:
    if libspnav.spnav_open() == -1:
        raise SpnavConnectionException("Failed to open a connection to the spacenav daemon.")


def spnav_fd() -> int:
    return libspnav.spnav_fd()


def spnav_close() -> None:
    libspnav.spnav_close()


def spnav_poll_event() -> SpnavEvent | None:
    event = spnav_event()
    if libspnav.spnav_poll_event(pointer(event)) == 0:
        return None
    return _convert_event(event)


def spnav_remove_events(event_type: int = SPNAV_EVENT_ANY) -> int:
    return libspnav.spnav_remove_events(event_type)
