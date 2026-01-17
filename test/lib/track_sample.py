import csv
from dataclasses import dataclass

TRACK_KEY_TOGGLE = ord("r")   # start/stop recording
TRACK_KEY_CLEAR  = ord("c")   # clear buffer
TRACK_KEY_WRITE  = ord("w")   # write file now

MIN_RECORD_HZ = 20.0          # ограничение частоты записи (защита от дублей)
MIN_RECORD_DT = 1.0 / MIN_RECORD_HZ

@dataclass
class TrackSample:
    ts_ns: int
    x: float
    y: float
    z: float
    qw: float
    qx: float
    qy: float
    qz: float
    tag_used: int
    held: bool
