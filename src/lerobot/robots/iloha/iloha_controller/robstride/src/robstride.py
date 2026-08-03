import asyncio
import collections
import logging
import math
import struct
import time
from asyncio import Lock
from dataclasses import dataclass
from logging import Formatter, StreamHandler, getLogger
from typing import Any, List, Optional, Union
import serial_asyncio

from .constants import CommandType, FaultCode, MotorStatus, ParameterIndex, RunMode

# Improved logger configuration
logger = getLogger(__name__)

# デバッグ設定: Trueにすると各制御サイクルで電圧のみを個別に読み取り、コンソールに表示します
DEBUG_VBUS_EVERY_CYCLE = False  # Trueにするとサイクルごとにバス負荷が2倍になるので通常はFalse
logger.setLevel(logging.ERROR)
handler_format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler = StreamHandler()
stream_handler.setLevel(logging.ERROR)
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)


@dataclass
class RobStrideLimits:
    """RobStrideモーターの制限パラメータを管理するクラス"""

    # PP (Profile Position) Mode limits
    pp_vel_max: Optional[float] = None  # 最大速度 (rad/s)
    pp_acc_set: Optional[float] = None  # 加速度 (rad/s^2)
    pp_limit_cur: Optional[float] = None  # 電流制限 (A)

    # Velocity Mode limits
    velocity_limit_cur: Optional[float] = None  # 電流制限 (A)
    velocity_acc_rad: Optional[float] = None  # 加速度 (rad/s^2)

    # CSP (Cyclic Synchronous Position) Mode limits
    csp_limit_spd: Optional[float] = None  # 速度制限 (rad/s)
    csp_limit_cur: Optional[float] = None  # 電流制限 (A)


@dataclass
class RobStride:
    id: int
    offset: float  # in radians
    limits: Optional[RobStrideLimits] = None
    _is_enabled: bool = False
    _current_mode: Optional[RunMode] = None

    def is_enabled(self) -> bool:
        """モーターの有効状態を返す"""
        return self._is_enabled

    def get_current_mode(self) -> Optional[RunMode]:
        """現在の制御モードを返す"""
        return self._current_mode

    def _set_enabled(self, enabled: bool) -> None:
        """モーターの有効状態を設定（内部使用）"""
        self._is_enabled = enabled

    def _set_mode(self, mode: Optional[RunMode]) -> None:
        """現在の制御モードを設定（内部使用）"""
        self._current_mode = mode


class RobStrideController:
    def __init__(
        self,
        port: str,
        motors: list[RobStride],
        baudrate: int = 921600,
        host_id: int = 253,
        log_timeout_errors: bool = True,
        log_latency_stats: bool = True,
    ):
        self.port = port
        self.baudrate = baudrate
        self.motors = {motor.id: motor for motor in motors}
        self.host_id = host_id
        self.log_timeout_errors = log_timeout_errors
        self.log_latency_stats = log_latency_stats
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.lock = Lock()
        # 診断用の統計情報
        self._latencies = collections.defaultdict(list)
        self._last_print_time = time.perf_counter()

    def _create_frame(
        self,
        command_type: CommandType,
        motor_id: int,
        data_area2: int = 0,
        data_payload: bytes = b'\x00' * 8,
    ) -> bytes:
        if command_type in [
            CommandType.GET_DEVICE_ID,
            CommandType.ENABLE,
            CommandType.DISABLE,
        ]:
            data_area2 = self.host_id

        can_id_29bit = (command_type.value << 24) | (data_area2 << 8) | motor_id
        encoded_id_32bit = (can_id_29bit << 3) | 0b100
        header = b'\x41\x54'
        encoded_id_bytes = encoded_id_32bit.to_bytes(4, 'big')
        extended_frame_flag = b'\x08'
        tail = b'\x0d\x0a'
        byte = header + encoded_id_bytes + extended_frame_flag + data_payload + tail
        assert isinstance(byte, bytes) and len(byte) == 17, "Frame must be 17 bytes"
        return byte

    async def _send_and_receive(self, frame: bytes) -> Optional[bytes]:
        # ★ このロックが単一バスの混線を防ぐ
        async with self.lock:
            if not self.writer or self.writer.is_closing():
                logger.error("Serial connection is not open")
                return None

            # 通信対象のモーターIDを抽出（デバッグ・診断用）
            encoded_id_32bit = int.from_bytes(frame[2:6], "big")
            motor_id = (encoded_id_32bit >> 3) & 0xFF

            start_time = time.perf_counter()
            try:
                # 1. Clear any pending data in the reader (flush)
                while not self.reader.at_eof():
                    try:
                        # Non-blocking check for internal buffer
                        await asyncio.wait_for(self.reader.read(1024), timeout=0.0001)
                    except asyncio.TimeoutError:
                        break

                # 2. Send the frame
                self.writer.write(frame)
                await self.writer.drain()

                # 3. バイナリpayload内のCRLFを終端と誤認しないよう、
                #    ATヘッダーへ同期して17バイト固定長で読む。
                response = await asyncio.wait_for(
                    self._read_response_frame(), timeout=0.015
                )

            except asyncio.TimeoutError:
                # Just log and return None without disabling the motor
                if self.log_timeout_errors:
                    logger.error(f"[Motor {motor_id}] No response received from motor within 15ms")
                return None
            except Exception as e:
                logger.error(f"[Motor {motor_id}] Error during serial I/O: {e}")
                return None
            finally:
                # 診断用統計の記録
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                
                self._latencies[motor_id].append(latency_ms)

                # 5秒おきに統計を表示
                if self.log_latency_stats and end_time - self._last_print_time > 5.0:
                    self._print_latency_stats()
                    self._last_print_time = end_time

            if response and response.startswith(b'AT') and response.endswith(b'\r\n'):
                # logger.debug(f"Received valid response: {response.hex(' ')}")
                if len(response) == 17:
                    return response
                else:
                    logger.error(
                        f"Invalid response length: expected 17 bytes, got {len(response)}"
                    )
            else:
                logger.error(
                    f"Invalid response format received: {response.hex(' ') if response else 'None'}"
                )
            return None

    async def _read_response_frame(self) -> bytes:
        assert self.reader is not None

        while True:
            if await self.reader.readexactly(1) != b"A":
                continue
            if await self.reader.readexactly(1) != b"T":
                continue
            response = b"AT" + await self.reader.readexactly(15)
            if response.endswith(b"\r\n"):
                return response

    def _print_latency_stats(self):
        """直近の通信遅延統計をターミナルに表示"""
        print(f"\n--- RobStride Latency Stats ({self.port}) ---")
        for motor_id, latencies in sorted(self._latencies.items()):
            if not latencies:
                continue
            avg_lat = sum(latencies) / len(latencies)
            max_lat = max(latencies)
            count = len(latencies)
            print(f" Motor {motor_id:3d} | Avg: {avg_lat:6.2f} ms | Max: {max_lat:6.2f} ms | Samples: {count:4d}")
            latencies.clear()  # 次回のためにクリア
        print("-------------------------------------------\n")

    async def _read_parameter(self, motor_id: int, index: int) -> Optional[bytes]:
        payload = struct.pack('<H', index) + b'\x00' * 6
        frame = self._create_frame(
            CommandType.READ_PARAM, motor_id, self.host_id, payload
        )
        response = await self._send_and_receive(frame)
        if response:
            return response[11:15]
        return None

    async def ping(self, motor_id: int) -> bool:
        """
        モーターとの通信が可能か確認する。
        
        Args:
            motor_id: 確認対象のモーターID
        Returns:
            bool: 通信成功ならTrue
        """
        # index 0x0000 (Mode) を読み取って疎通確認
        res = await self._read_parameter(motor_id, 0x0000)
        return res is not None

    def decode_fault_code(self, code: int) -> List[str]:
        """故障コードを人間が読みやすい文字列のリストに変換"""
        if code == 0:
            return ["None"]
        
        faults = []
        for fault in FaultCode:
            if fault != FaultCode.NONE and (code & fault.value):
                # Enum名のアンダースコアをスペースに置換して読みやすく
                faults.append(fault.name.replace("_", " ").title())
        
        if not faults:
            return [f"Unknown Fault (0x{code:08X})"]
        return faults

    async def get_parameter(
        self,
        motor_id: int,
        index: ParameterIndex,
        data_type: str = "float"
    ) -> Optional[Union[float, int]]:
        """
        モーターから特定のパラメータを読み出す。
        
        Args:
            motor_id: モーターID
            index: ParameterIndex
            data_type: "float", "uint32", "uint16", "uint8"
        """
        raw_bytes = await self._read_parameter(motor_id, index.value)
        if raw_bytes is None:
            return None
        
        try:
            if data_type == "float":
                # float (4 bytes, little endian)
                return struct.unpack('<f', raw_bytes)[0]
            elif data_type == "uint32":
                # uint32 (4 bytes, little endian)
                return struct.unpack('<I', raw_bytes)[0]
            elif data_type == "uint16":
                # uint16 (2 bytes, checking both positions as it might be in Type 2 payload)
                val = struct.unpack('<H', raw_bytes[0:2])[0]
                # If value is 0 but it's a temperature register, it might be in the second half
                if val == 0 and len(raw_bytes) >= 4:
                    val = struct.unpack('<H', raw_bytes[2:4])[0]
                return val
            elif data_type == "int16":
                # int16 (2 bytes, checking both positions)
                val = struct.unpack('<h', raw_bytes[0:2])[0]
                if val == 0 and len(raw_bytes) >= 4:
                    val = struct.unpack('<h', raw_bytes[2:4])[0]
                return val
            elif data_type == "int32":
                # int32 (4 bytes, little endian)
                return struct.unpack('<i', raw_bytes)[0]
            elif data_type == "uint8":
                # uint8 (1 byte at the beginning)
                return raw_bytes[0]
            else:
                logger.error(f"Unsupported data type: {data_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to decode parameter {index.name}: {e}")
            return None

    async def get_motor_status_comprehensive(self, motor_id: int) -> Optional[dict]:
        """
        モーターの全パラメータ（主要なもの）を読み出し、デコード済み辞書として返す
        """
        if motor_id not in self.motors:
            return None

        status = {"motor_id": motor_id}
        
        # 1. リアルタイムフィードバックから温度、トルク等を取得
        feedback = await self.get_motor_feedback(motor_id)
        if feedback:
            status["motor_temp"] = feedback["temp"]
            status["torque_fdb"] = feedback["torque"]
            # 注意: Feedbackフレームのトルクは生値なので必要に応じてスケーリングが必要な場合がありますが、
            # 現状はそのまま格納します
        
        # 2. 読み取るパラメータとその型のリスト
        params_to_read = [
            # 制御モード・目標
            (ParameterIndex.RUN_MODE, "uint8"),
            (ParameterIndex.LOC_REF, "float"),
            (ParameterIndex.SPD_REF, "float"),
            (ParameterIndex.IQ_REF, "float"),
            
            # 実測値
            (ParameterIndex.MECH_POS, "float"),
            (ParameterIndex.MECH_VELO, "float"),
            (ParameterIndex.VBUS, "float"),
            (ParameterIndex.IQF, "float"),
            
            # 制限設定
            (ParameterIndex.LIMIT_TORQUE, "float"),
            (ParameterIndex.LIMIT_SPD, "float"),
            (ParameterIndex.LIMIT_CUR, "float"),
            (ParameterIndex.VEL_MAX, "float"),
            (ParameterIndex.ACC_SET, "float"),
            (ParameterIndex.ACC_RAD, "float"),
            
            # ゲイン設定
            (ParameterIndex.LOC_KP, "float"),
            (ParameterIndex.SPD_KP, "float"),
            (ParameterIndex.SPD_KI, "float"),
            (ParameterIndex.SPD_FILT_GAIN, "float"),
            (ParameterIndex.CUR_KP, "float"),
            (ParameterIndex.CUR_KI, "float"),
            (ParameterIndex.CUR_FILT_GAIN, "float"),
            
            # システム・通信
            (ParameterIndex.CAN_TIMEOUT, "uint32"),
            (ParameterIndex.EPSCAN_TIME, "uint16"),
            (ParameterIndex.ZERO_STA, "uint8"),
            (ParameterIndex.DAMPER, "uint8"),
            (ParameterIndex.ADD_OFFSET, "float"),
            
            # 故障診断
            (ParameterIndex.FAULT_CODE, "uint32"),

            # --- 内部詳細計追加 ---
            # MOTOR_TEMP と TORQUE_FDB はフィードバックフレームから取得済みのためここからは除外
            (ParameterIndex.MCU_TEMP, "int16"),
            (ParameterIndex.BOARD_TEMP, "int16"),
            (ParameterIndex.DRV_TEMP, "int16"),
            (ParameterIndex.ID_RAW, "float"),
            (ParameterIndex.IQ_RAW, "float"),
            (ParameterIndex.DRV_FAULT, "uint16"),
        ]

        async def read_and_store(p_idx, d_type):
            val = await self.get_parameter(motor_id, p_idx, d_type)
            if val is not None:
                # 温度パラメータは 10倍 されているので 0.1倍 する
                temp_indices = [
                    ParameterIndex.MOTOR_TEMP,
                    ParameterIndex.MCU_TEMP,
                    ParameterIndex.BOARD_TEMP,
                    ParameterIndex.DRV_TEMP
                ]
                if p_idx in temp_indices:
                    val = val / 10.0
                
                if p_idx == ParameterIndex.FAULT_CODE:
                    status["fault_code_raw"] = val
                    status["fault_list"] = self.decode_fault_code(val)
                elif p_idx == ParameterIndex.RUN_MODE:
                    try:
                        status["run_mode"] = RunMode(val).name
                    except ValueError:
                        status["run_mode"] = f"Unknown ({val})"
                else:
                    status[p_idx.name.lower()] = val

        # 順次読み込み（Lock競合を考慮）
        for p_idx, d_type in params_to_read:
            await read_and_store(p_idx, d_type)
            
        return status

    async def get_version(self, motor_id: int) -> Optional[dict]:
        """モーターのハードウェア/ソフトウェアバージョンを取得 (Type 26)"""
        frame = self._create_frame(CommandType.GET_VERSION, motor_id)
        response = await self._send_and_receive(frame)
        if response and len(response) >= 15:
            # ペイロードは response[7:15]
            # Byte 0~3: ハードウェアバージョン (Big-endian)
            # Byte 4~7: ソフトウェアバージョン (Big-endian)
            # マニュアルの例: 0x01000000 -> V1.0.0.0
            hw_v = response[7:11]
            sw_v = response[11:15]
            
            def fmt(b):
                return f"V{b[0]}.{b[1]}.{b[2]}.{b[3]}"
                
            return {
                "hw": fmt(hw_v),
                "sw": fmt(sw_v)
            }
        return None

    async def _write_parameter(
        self, motor_id: int, index: int, value: Union[int, float]
    ) -> Optional[bytes]:
        if isinstance(value, int):
            # For RunMode, which is a uint8, pack as a 4-byte integer
            payload = struct.pack('<H', index) + b'\x00\x00' + struct.pack('<I', value)
        elif isinstance(value, float):
            payload = struct.pack('<H', index) + b'\x00\x00' + struct.pack('<f', value)
        else:
            return None

        frame = self._create_frame(
            CommandType.WRITE_PARAM, motor_id, self.host_id, payload
        )
        return await self._send_and_receive(frame)

    async def connect(self) -> bool:
        logger.info("Initiating connection to motors")
        try:
            # Create serial connection with asyncio
            coro = serial_asyncio.open_serial_connection(
                url=self.port, baudrate=self.baudrate, timeout=1.0
            )
            self.reader, self.writer = await coro
            logger.info(f"Serial port {self.port} opened successfully")
        except Exception as e:
            logger.error(f"Failed to open serial port {self.port}: {e}")
            return False

        # Check connection to all motors
        all_connected = True
        for motor_id in self.motors.keys():
            frame = self._create_frame(CommandType.GET_DEVICE_ID, motor_id)
            if await self._send_and_receive(frame):
                logger.info(
                    f"Connection established successfully with motor ID {motor_id}"
                )
            else:
                logger.error(f"Failed to establish connection with motor ID {motor_id}")
                all_connected = False

        if all_connected:
            logger.info("All motors connected successfully")
            return True
        else:
            logger.error("Failed to connect to some motors")
            await self.disconnect()
            return False

    async def enable(self, motor_id: int) -> bool:
        if motor_id not in self.motors:
            logger.error(f"Motor ID {motor_id} not found in motor list")
            return False

        logger.info(f"Enabling motor {motor_id}")
        frame = self._create_frame(CommandType.ENABLE, motor_id)
        response = await self._send_and_receive(frame)
        if not response:
            logger.error(f"Failed to send enable command to motor {motor_id}")
            return False

        can_id_29bit = int.from_bytes(response[2:6], 'big') >> 3
        status_val = (can_id_29bit >> 22) & 0b11
        status = (
            MotorStatus(status_val)
            if status_val in [m.value for m in MotorStatus]
            else MotorStatus.UNKNOWN
        )

        if status == MotorStatus.RUN:
            logger.info(f"Motor {motor_id} enabled successfully and entered RUN state")
            self.motors[motor_id]._set_enabled(True)
            return True
        else:
            logger.error(
                f"Motor {motor_id} enable failed: Invalid status {status.name}"
            )
            return False

    async def disable(self, motor_id: int) -> bool:
        """指定されたモーターを無効化（運転停止）します。"""
        if motor_id not in self.motors:
            logger.error(f"Motor ID {motor_id} not found in motor list")
            return False

        logger.info(f"Disabling motor {motor_id}")
        frame = self._create_frame(CommandType.DISABLE, motor_id)
        response = await self._send_and_receive(frame)
        if response is None:
            logger.error(f"Failed to disable motor {motor_id}")
            return False
        self.motors[motor_id]._set_enabled(False)
        self.motors[motor_id]._set_mode(None)
        logger.info(f"Disable command sent successfully to motor {motor_id}")
        return True

    async def save_parameters(self, motor_id: int) -> bool:
        """
        現在のパラメータ設定をモーターの不揮発メモリ（フラッシュ）に保存します。
        書き換え回数に制限があるため、設定変更時のみ呼び出してください。
        """
        if motor_id not in self.motors:
            logger.error(f"Motor ID {motor_id} not found in motor list")
            return False

        logger.info(f"Saving parameters to NVM for motor {motor_id}")
        # 仕様書に基づき、ペイロードに 01 02 03 04 05 06 07 08 を設定
        payload = b"\x01\x02\x03\x04\x05\x06\x07\x08"
        frame = self._create_frame(CommandType.SAVE_PARAM, motor_id, data_payload=payload)
        response = await self._send_and_receive(frame)
        
        if response is not None:
            logger.info(f"Save parameters successful for motor {motor_id}")
            return True
        else:
            logger.error(f"Save parameters failed for motor {motor_id}")
            return False

    def _check_motor_enabled(self, motor_id: int) -> bool:
        """モーターが有効かどうかをチェック"""
        if motor_id not in self.motors:
            logger.error(f"Motor ID {motor_id} not found in motor list")
            return False

        if not self.motors[motor_id].is_enabled():
            logger.error(
                f"Motor {motor_id} is not enabled. Please enable the motor first."
            )
            return False

        return True

    def _check_motor_mode(
        self,
        motor_id: int,
        required_mode: RunMode,
        *,
        allow_disabled: bool = False,
    ) -> bool:
        """モーターが指定されたモードかどうかをチェック"""
        if motor_id not in self.motors:
            logger.error(f"Motor ID {motor_id} not found in motor list")
            return False

        if not allow_disabled and not self._check_motor_enabled(motor_id):
            return False

        current_mode = self.motors[motor_id].get_current_mode()
        if current_mode != required_mode:
            logger.error(
                f"Motor {motor_id} is not in {required_mode.name} mode. Current mode: {current_mode.name if current_mode else 'None'}"
            )
            return False

        return True

    async def _set_run_mode(self, motor_id: int, mode: RunMode) -> bool:
        logger.info(f"Setting motor {motor_id} to {mode.name} mode")
        await self._write_parameter(motor_id, ParameterIndex.RUN_MODE.value, mode.value)

        await asyncio.sleep(0.1)
        read_data = await self._read_parameter(motor_id, ParameterIndex.RUN_MODE.value)
        if read_data:
            current_mode = int.from_bytes(read_data[0:1], 'little')
            if current_mode == mode.value:
                logger.info(f"Motor {motor_id} {mode.name} mode set successfully")
                self.motors[motor_id]._set_mode(mode)
                return True
            else:
                logger.error(
                    f"Motor {motor_id} {mode.name} mode setting failed: Unexpected run_mode value {current_mode}"
                )
        else:
            logger.error(f"Failed to read run_mode parameter for motor {motor_id}")
        return False

    async def _set_float_parameter(
        self,
        motor_id: int,
        param_index: ParameterIndex,
        value: float,
        name: str,
        unit: str,
    ) -> bool:
        if motor_id not in self.motors:
            logger.error(f"Motor ID {motor_id} not found in motor list")
            return False

        logger.info(f"Setting {name} to {value} {unit} for motor {motor_id}")
        await self._write_parameter(motor_id, param_index.value, value)

        await asyncio.sleep(0.1)
        read_data = await self._read_parameter(motor_id, param_index.value)
        if read_data:
            current_val = struct.unpack('<f', read_data)[0]
            if math.isclose(current_val, value, rel_tol=1e-6):
                logger.info(
                    f"Motor {motor_id} {name} set successfully to {current_val:.2f} {unit}"
                )
                return True
            else:
                logger.error(
                    f"Motor {motor_id} {name} setting failed: Expected {value:.2f}, got {current_val:.2f} {unit}"
                )
        else:
            logger.error(f"Failed to read {name} parameter for motor {motor_id}")
        return False

    # --- PP (Profile Position) Mode Methods ---
    async def set_mode_pp(self, motor_id: int) -> bool:
        return await self._set_run_mode(motor_id, RunMode.POSITION_PP)

    async def apply_pp_limits(
        self,
        motor_id: int,
        *,
        allow_disabled: bool = False,
    ) -> bool:
        """PP モードのリミッターを適用"""
        if not self._check_motor_mode(
            motor_id,
            RunMode.POSITION_PP,
            allow_disabled=allow_disabled,
        ):
            return False

        motor = self.motors[motor_id]
        if not motor.limits:
            logger.warning(f"No limits configured for motor {motor_id}")
            return True

        success = True
        limits = motor.limits

        if limits.pp_vel_max is not None:
            success &= await self._set_float_parameter(
                motor_id,
                ParameterIndex.VEL_MAX,
                limits.pp_vel_max,
                "PP velocity",
                "rad/s",
            )

        if limits.pp_acc_set is not None:
            success &= await self._set_float_parameter(
                motor_id,
                ParameterIndex.ACC_SET,
                limits.pp_acc_set,
                "PP acceleration",
                "rad/s^2",
            )

        if limits.pp_limit_cur is not None:
            success &= await self._set_float_parameter(
                motor_id,
                ParameterIndex.LIMIT_CUR,
                limits.pp_limit_cur,
                "PP current limit",
                "A",
            )

        return success

    async def set_target_position(
        self, motor_id: int, position_rad: float
    ) -> Optional[bytes]:
        if motor_id not in self.motors:
            logger.error(f"Motor ID {motor_id} not found in motor list")
            return

        logger.info(
            f"Setting target position to {position_rad:.2f} rad for motor {motor_id}"
        )
        # --- 電圧デバッグスパム ---
        if DEBUG_VBUS_EVERY_CYCLE:
            vbus = await self.get_parameter(motor_id, ParameterIndex.VBUS, "float")
            if vbus is not None:
                print(f"DEBUG [Motor {motor_id}] VBUS: {vbus:.2f}V")
        # ------------------------

        target_pos_rad = position_rad + self.motors[motor_id].offset
        result = await self._write_parameter(
            motor_id, ParameterIndex.LOC_REF.value, target_pos_rad
        )
        if result is None:
            # Trigger diagnostic dump on failure
            await self.log_motor_diagnostics(motor_id, reason="Communication Error / Timeout")
            logger.error(f"Failed to send target position to motor {motor_id}")
        return result

    # --- Velocity Mode Methods ---
    async def set_mode_velocity(self, motor_id: int) -> bool:
        return await self._set_run_mode(motor_id, RunMode.VELOCITY)

    async def apply_velocity_limits(self, motor_id: int) -> bool:
        """Velocity モードのリミッターを適用"""
        if not self._check_motor_mode(motor_id, RunMode.VELOCITY):
            return False

        motor = self.motors[motor_id]
        if not motor.limits:
            logger.warning(f"No limits configured for motor {motor_id}")
            return True

        success = True
        limits = motor.limits

        if limits.velocity_limit_cur is not None:
            success &= await self._set_float_parameter(
                motor_id,
                ParameterIndex.LIMIT_CUR,
                limits.velocity_limit_cur,
                "Velocity current limit",
                "A",
            )

        if limits.velocity_acc_rad is not None:
            success &= await self._set_float_parameter(
                motor_id,
                ParameterIndex.ACC_RAD,
                limits.velocity_acc_rad,
                "Velocity acceleration",
                "rad/s^2",
            )

        return success

    async def set_target_velocity(self, motor_id: int, velocity: float) -> None:
        """速度制御モードで目標速度を設定します。"""
        if motor_id not in self.motors:
            logger.error(f"Motor ID {motor_id} not found in motor list")
            return

        logger.info(
            f"Setting target velocity to {velocity:.2f} rad/s for motor {motor_id}"
        )
        await self._write_parameter(motor_id, ParameterIndex.SPD_REF.value, velocity)
        logger.info(f"Target velocity command sent successfully to motor {motor_id}")

    # --- Current Mode Methods ---
    async def set_mode_current(self, motor_id: int) -> bool:
        return await self._set_run_mode(motor_id, RunMode.CURRENT)

    async def set_target_current(self, motor_id: int, current: float) -> None:
        if motor_id not in self.motors:
            logger.error(f"Motor ID {motor_id} not found in motor list")
            return

        logger.info(f"Setting target current to {current:.2f} A for motor {motor_id}")
        await self._write_parameter(motor_id, ParameterIndex.IQ_REF.value, current)
        logger.info(f"Target current command sent successfully to motor {motor_id}")

    # --- CSP (Cyclic Synchronous Position) Mode Methods ---
    async def set_mode_csp(self, motor_id: int) -> bool:
        return await self._set_run_mode(motor_id, RunMode.POSITION_CSP)

    async def apply_csp_limits(
        self,
        motor_id: int,
        *,
        allow_disabled: bool = False,
    ) -> bool:
        """CSP モードのリミッターを適用"""
        if not self._check_motor_mode(
            motor_id,
            RunMode.POSITION_CSP,
            allow_disabled=allow_disabled,
        ):
            return False

        motor = self.motors[motor_id]
        if not motor.limits:
            logger.warning(f"No limits configured for motor {motor_id}")
            return True

        success = True
        limits = motor.limits

        if limits.csp_limit_spd is not None:
            success &= await self._set_float_parameter(
                motor_id,
                ParameterIndex.LIMIT_SPD,
                limits.csp_limit_spd,
                "CSP velocity limit",
                "rad/s",
            )

        if limits.csp_limit_cur is not None:
            success &= await self._set_float_parameter(
                motor_id,
                ParameterIndex.LIMIT_CUR,
                limits.csp_limit_cur,
                "CSP current limit",
                "A",
            )

        return success

    async def disconnect(self) -> None:
        if self.writer and not self.writer.is_closing():
            self.writer.close()
            await self.writer.wait_closed()
            logger.info("Serial port closed")

    async def __aenter__(self) -> 'RobStrideController':
        """async with構文の開始時に接続を行います。"""
        if await self.connect():
            return self
        else:
            raise IOError("Failed to establish connection with motor")

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """async with構文の終了時に、安全にモーターを停止し、切断します。"""
        if self.writer and not self.writer.is_closing():
            logger.info(f"Safely shutting down motors on {self.port} sequentially...")
            for motor_id in self.motors.keys():
                await self.set_target_velocity(motor_id, 0.0)
                await self.set_target_current(motor_id, 0.0)
            await asyncio.sleep(0.1)
            for motor_id in self.motors.keys():
                await self.disable(motor_id)
        await self.disconnect()

    def __enter__(self) -> 'RobStrideController':
        raise NotImplementedError("Use 'async with' instead of 'with'")

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        raise NotImplementedError("Use 'async with' instead of 'with'")

    async def get_motor_feedback(self, motor_id: int) -> Optional[dict]:
        """
        Communication Type 1 (GET_STATUS) を送り、
        Type 2 (Feedback Frame) からリアルタイム情報を取得する
        """
        frame = self._create_frame(
            CommandType.GET_STATUS, motor_id, self.host_id, b'\x00' * 8
        )
        response = await self._send_and_receive(frame)
        if response and len(response) == 17:
            payload = response[7:15]
            try:
                # Type 2 Feedback Frame uses mapped uint16 for most values
                # according to RobStride/CyberGear protocol.
                angle_raw = struct.unpack('>H', payload[0:2])[0]
                velocity_raw = struct.unpack('>H', payload[2:4])[0]
                torque_raw = struct.unpack('>H', payload[4:6])[0]
                temp_raw = struct.unpack('>H', payload[6:8])[0]
                
                # Conversion functions
                def uint_to_float(x, min_v, max_v):
                    return min_v + (max_v - min_v) * x / 65535.0

                import math
                return {
                    "angle": uint_to_float(angle_raw, -4 * math.pi, 4 * math.pi),
                    "velocity": uint_to_float(velocity_raw, -30.0, 30.0),
                    "torque": uint_to_float(torque_raw, -12.0, 12.0),
                    "temp": temp_raw / 10.0
                }
            except Exception as e:
                logger.error(f"Failed to parse feedback frame for motor {motor_id}: {e}")
        return None

    async def log_motor_diagnostics(self, motor_id: int, reason: str = "Diagnostic"):
        """
        モーターの内部状態を読み出し、詳細な診断ログを出力する (レート制限付き)
        """
        now = time.time()
        if not hasattr(self, "_last_diag_time"):
            self._last_diag_time = {}
        if now - self._last_diag_time.get(motor_id, 0) < 3.0:
            return

        self._last_diag_time[motor_id] = now
        
        logger.warning(f"🔍 [DIAGNOSTIC] Motor {motor_id} Error detected. Reason: {reason}")
        
        try:
            feedback = await self.get_motor_feedback(motor_id)
            fault_sta = await self.get_parameter(motor_id, ParameterIndex.FAULT_STA, "uint32")
            drv_fault = await self.get_parameter(motor_id, ParameterIndex.DRV_FAULT, "uint16")
            vbus = await self.get_parameter(motor_id, ParameterIndex.VBUS, "float")
            mcu_temp_reg = await self.get_parameter(motor_id, ParameterIndex.MCU_TEMP, "int16")
            
            m_temp = feedback["temp"] if feedback else "N/A"
            torque = feedback["torque"] if feedback else "N/A"
            mcu_temp = mcu_temp_reg / 10.0 if mcu_temp_reg is not None and mcu_temp_reg != 0 else "N/A"
            
            fault_desc = "None"
            if fault_sta:
                from .constants import FaultCode
                active_faults = [f.name for f in FaultCode if fault_sta & f.value]
                fault_desc = ", ".join(active_faults) if active_faults else f"Unknown ({hex(fault_sta)})"

            diag_msg = (
                f"\n"
                f"╔════════════════════════════════════════════════════════════╗\n"
                f"║ [ MOTOR {motor_id} DIAGNOSTIC REPORT ]\n"
                f"╟────────────────────────────────────────────────────────────╢\n"
                f"║  - Error Reason : {reason}\n"
                f"║  - Temp (Motor) : {m_temp}℃\n"
                f"║  - Temp (MCU)   : {mcu_temp}℃\n"
                f"║  - Bus Voltage  : {vbus if vbus is not None else 'N/A'}V\n"
                f"║  - Raw Torque   : {torque}\n"
                f"║  - Fault Status : {fault_desc}\n"
                f"║  - Fault Raw    : {hex(fault_sta) if fault_sta is not None else 'N/A'}\n"
                f"║  - Driver Fault : {drv_fault if drv_fault is not None else 'N/A'}\n"
                f"╚════════════════════════════════════════════════════════════╝\n"
            )
            print(diag_msg)
            
        except Exception as e:
            logger.error(f"Failed to retrieve diagnostics for motor {motor_id}: {e}")
