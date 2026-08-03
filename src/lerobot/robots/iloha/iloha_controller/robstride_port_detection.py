import asyncio
import logging
from dataclasses import dataclass

from serial.tools import list_ports

from .robstride.src.robstride import RobStrideController

AUTO_PORT = "auto"
LEFT_ROBSTRIDE_IDS = (1, 2, 3)
RIGHT_ROBSTRIDE_IDS = (4, 5, 6)
ROBSTRIDE_USB_ADAPTER_IDS = {(0x1A86, 0x7523)}
ROBSTRIDE_DETECTION_NUM_RETRY = 3

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RobStridePorts:
    left: str
    right: str


def _find_candidate_ports() -> list[str]:
    """接続中のRobStride用CH340 USBシリアルポートを列挙する。"""
    return sorted(
        port.device
        for port in list_ports.comports()
        if (port.vid, port.pid) in ROBSTRIDE_USB_ADAPTER_IDS
    )


async def _all_motors_respond(
    controller: RobStrideController,
    motor_ids: tuple[int, ...],
) -> bool:
    for motor_id in motor_ids:
        for _ in range(1 + ROBSTRIDE_DETECTION_NUM_RETRY):
            if await controller.ping(motor_id):
                break
        else:
            return False
    return True


async def _classify_port(port: str) -> str | None:
    """モータIDへの読取り応答から、ポートが左右どちらかを判定する。"""
    controller = RobStrideController(
        port=port,
        motors=[],
        log_timeout_errors=False,
        log_latency_stats=False,
    )
    try:
        if not await controller.connect():
            return None
        if await _all_motors_respond(controller, LEFT_ROBSTRIDE_IDS):
            return "left"
        if await _all_motors_respond(controller, RIGHT_ROBSTRIDE_IDS):
            return "right"
        return None
    except Exception as exc:
        logger.warning("RobStride port probe failed on %s: %s", port, exc)
        return None
    finally:
        await controller.disconnect()


async def detect_robstride_ports() -> RobStridePorts:
    """RobStrideの左右ポートをモータIDへの応答から自動検出する。"""
    candidates = _find_candidate_ports()
    if not candidates:
        raise RuntimeError(
            "RobStride用CH340ポートが見つかりません。USB接続と電源を確認してください。"
        )

    classifications = await asyncio.gather(*(_classify_port(port) for port in candidates))
    matches = {
        side: [
            port
            for port, detected_side in zip(candidates, classifications, strict=True)
            if detected_side == side
        ]
        for side in ("left", "right")
    }
    if len(matches["left"]) != 1 or len(matches["right"]) != 1:
        results = dict(zip(candidates, classifications, strict=True))
        raise RuntimeError(
            "RobStrideポートを一意に判定できませんでした。"
            f" expected left IDs={LEFT_ROBSTRIDE_IDS}, "
            f"right IDs={RIGHT_ROBSTRIDE_IDS}, results={results}"
        )

    return RobStridePorts(left=matches["left"][0], right=matches["right"][0])


async def resolve_robstride_ports(left: str, right: str) -> RobStridePorts:
    """`auto` が指定された側だけを自動検出結果で置き換える。"""
    if left != AUTO_PORT and right != AUTO_PORT:
        return RobStridePorts(left=left, right=right)

    detected = await detect_robstride_ports()
    return RobStridePorts(
        left=detected.left if left == AUTO_PORT else left,
        right=detected.right if right == AUTO_PORT else right,
    )
