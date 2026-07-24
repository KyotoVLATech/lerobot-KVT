from dataclasses import dataclass

from lerobot.motors.dynamixel import OperatingMode


@dataclass
class Robstride01Constants:
    """根本のRobStrideの定数"""

    ID = 1
    DEFAULT_OFFSET = 0.0  # デフォルトオフセット [rad]


@dataclass
class Robstride02Constants:
    """2番目のRobStrideの定数"""

    ID = 2
    DEFAULT_OFFSET = 0.0  # デフォルトオフセット [rad]


@dataclass
class Robstride03Constants:
    """3番目のRobStrideの定数"""

    ID = 3
    DEFAULT_OFFSET = 0.0  # デフォルトオフセット [rad]


@dataclass
class Dynamixel01Constants:
    """根本のDynamixelの定数"""

    MODEL = "xm540-w270"
    ID = 1
    OPERATING_MODE = OperatingMode.EXTENDED_POSITION
    OFFSET = 4096 // 2


@dataclass
class Dynamixel02Constants:
    """2番目のDynamixelの定数"""

    MODEL = "xm540-w270"
    ID = 2
    OPERATING_MODE = OperatingMode.POSITION
    OFFSET = 4096 // 2


@dataclass
class Dynamixel03Constants:
    """3番目のDynamixelの定数"""

    MODEL = "xm430-w350"
    ID = 3
    OPERATING_MODE = OperatingMode.POSITION
    OFFSET = 4096 // 2


@dataclass
class Dynamixel04Constants:
    """4番目のDynamixelの定数"""

    MODEL = "xm430-w350"
    ID = 4
    OPERATING_MODE = OperatingMode.CURRENT_POSITION
    OFFSET = 4096 // 2
