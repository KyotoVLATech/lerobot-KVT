import asyncio
import struct
import unittest
from unittest.mock import AsyncMock

from lerobot.robots.iloha.iloha_controller.robstride.src.constants import ParameterIndex
from lerobot.robots.iloha.iloha_controller.robstride.src.robstride import (
    RobStride,
    RobStrideController,
)


class TestFloatParameterEncoding(unittest.TestCase):
    def test_float_registers_encode_integer_settings_as_floats(self):
        for index in (ParameterIndex.ACC_SET, ParameterIndex.VEL_MAX, ParameterIndex.LIMIT_CUR):
            for value in (10, 10.0, 1.5):
                with self.subTest(index=index, value=value):
                    controller = RobStrideController("unused", [RobStride(1, 0.0)])
                    written = []

                    async def exchange(frame):
                        written.append(frame)
                        return b"ok"

                    async def read_back(motor_id, parameter_index):
                        # Emulate the motor reading the actual transmitted bytes.
                        return written[-1][11:15]

                    controller._send_and_receive = exchange
                    controller._read_parameter = read_back
                    result = asyncio.run(
                        controller._set_float_parameter(1, index, value, "limit", "")
                    )
                    self.assertTrue(result)
                    self.assertEqual(written[0][7:9], struct.pack("<H", index.value))
                    self.assertEqual(written[0][11:15], struct.pack("<f", value))

    def test_float_register_readback_mismatch_still_fails(self):
        controller = RobStrideController("unused", [RobStride(1, 0.0)])
        controller._send_and_receive = AsyncMock(return_value=b"ok")
        controller._read_parameter = AsyncMock(return_value=struct.pack("<f", 0.0))
        with self.assertLogs(level="ERROR"):
            self.assertFalse(asyncio.run(
                controller._set_float_parameter(1, ParameterIndex.ACC_SET, 10, "acceleration", "rad/s^2")
            ))

    def test_integer_register_preserves_integer_encoding(self):
        controller = RobStrideController("unused", [RobStride(1, 0.0)])
        controller._send_and_receive = AsyncMock(return_value=b"ok")
        asyncio.run(controller._write_parameter(1, ParameterIndex.RUN_MODE.value, 1))
        frame = controller._send_and_receive.call_args.args[0]
        self.assertEqual(frame[11:15], struct.pack("<I", 1))
