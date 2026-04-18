import enum


class CommandType(enum.Enum):
    """モーターへ送信するコマンドの種類"""

    GET_DEVICE_ID = 0x00
    GET_STATUS = 0x01
    ENABLE = 0x03
    DISABLE = 0x04
    READ_PARAM = 0x11
    WRITE_PARAM = 0x12
    SAVE_PARAM = 0x16
    GET_VERSION = 0x1A


class MotorStatus(enum.Enum):
    """モーターからの応答に含まれる状態"""

    RESET = 0
    CALIBRATION = 1
    RUN = 2
    UNKNOWN = 99


class RunMode(enum.Enum):
    """モーターの運転モード"""

    OPERATION = 0
    POSITION_PP = 1
    VELOCITY = 2
    CURRENT = 3
    POSITION_CSP = 5


class ParameterIndex(enum.Enum):
    """モーターパラメータのインデックス"""

    # --- 共通・システム (0x7000 series) ---
    RUN_MODE = 0x7005
    IQ_REF = 0x7006
    SPD_REF = 0x700A
    LIMIT_TORQUE = 0x700B
    CUR_KP = 0x7010
    CUR_KI = 0x7011
    CUR_FILT_GAIN = 0x7014
    LOC_REF = 0x7016
    LIMIT_SPD = 0x7017
    LIMIT_CUR = 0x7018
    MECH_POS = 0x7019
    IQF = 0x701A
    MECH_VELO = 0x701B
    VBUS = 0x701C
    LOC_KP = 0x701E
    SPD_KP = 0x701F
    SPD_KI = 0x7020
    SPD_FILT_GAIN = 0x7021
    ACC_RAD = 0x7022
    VEL_MAX = 0x7024
    ACC_SET = 0x7025
    EPSCAN_TIME = 0x7026
    CAN_TIMEOUT = 0x7028
    ZERO_STA = 0x7029
    DAMPER = 0x702A
    ADD_OFFSET = 0x702B
    FAULT_CODE = 0x702E

    # --- 内部状態 (0x3000 series Read Only) ---
    MCU_TEMP = 0x3005       # int16, *10
    MOTOR_TEMP = 0x3006     # int16, *10
    VBUS_MV = 0x3007        # uint16, mV
    CMD_ID = 0x300d         # float, A
    CMD_IQ = 0x300e         # float, A
    IA = 0x3019             # float, A
    IB = 0x301a             # float, A
    IC = 0x301b             # float, A
    BOARD_TEMP = 0x301f     # int16, *10
    IQ_RAW = 0x3020         # float, A
    ID_RAW = 0x3021         # float, A
    FAULT_STA = 0x3022      # uint32, Fault status
    WARN_STA = 0x3023       # uint32, Warning status
    DRV_FAULT = 0x3024      # uint16, Driver chip fault 1
    DRV_TEMP = 0x3025       # int16, *10, Driver chip fault 2/Temp
    UQ = 0x3026             # float, Q-axis voltage
    UD = 0x3027             # float, D-axis voltage
    TORQUE_FDB = 0x302c     # float, Nm


class FaultCode(enum.IntFlag):
    """故障コードのビットフラグ (0x3022)"""

    NONE = 0
    MOTOR_OVER_TEMP = 1 << 0  # モーター過熱 (>145℃)
    DRIVER_CHIP_FAULT = 1 << 1  # ドライバチップ故障
    UNDER_VOLTAGE = 1 << 2  # 不足電圧 (<12V)
    OVER_VOLTAGE = 1 << 3  # 過電圧 (>60V)
    B_PHASE_OVER_CURRENT = 1 << 4  # B相過電流
    C_PHASE_OVER_CURRENT = 1 << 5  # C相過電流
    ENCODER_NOT_CALIBRATED = 1 << 7  # エンコーダ未校正
    HARDWARE_ID_FAULT = 1 << 8  # ハードウェアID故障
    POSITION_INIT_FAULT = 1 << 9  # 位置初期化故障
    STALL_OVERLOAD = 1 << 14  # 脱調・過負荷保護
    A_PHASE_OVER_CURRENT = 1 << 16  # A相過電流
