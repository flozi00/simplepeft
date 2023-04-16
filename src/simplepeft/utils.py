from enum import Enum
import platform

os = platform.system()

IS_WINDOWS = (os == "Windows")


class Tasks(Enum):
    ASR = 1
    Text2Text = 2
    TEXT_GEN = 3
