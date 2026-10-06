#!/usr/bin/env python3
"""Inspect or gently jog ONLY configured right Dynamixel gripper (ID 8).

No Iloha/Aloha initialization, RobStride access, broadcasts, or homing.
Default is read-only. --move additionally requires interactive confirmation.
"""

import argparse
import time

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler


MOTOR_ID = 8
PORT = "/dev/ttyUSB_RightDynamixel"


class Gripper:
    def __init__(self, port):
        self.port = PortHandler(port)
        self.packet = PacketHandler(2.0)

    def check(self, result, error):
        if result != COMM_SUCCESS:
            raise RuntimeError(self.packet.getTxRxResult(result))
        if error:
            raise RuntimeError(self.packet.getRxPacketError(error))

    def read(self, address, size):
        value, result, error = getattr(self.packet, f"read{size}ByteTxRx")(
            self.port, MOTOR_ID, address
        )
        self.check(result, error)
        return value

    def write(self, address, size, value):
        result, error = getattr(self.packet, f"write{size}ByteTxRx")(
            self.port, MOTOR_ID, address, value
        )
        self.check(result, error)

    def position(self):
        value = self.read(132, 4)
        return value - 2**32 if value >= 2**31 else value

    def move_to(self, target):
        print(f"ID 8のみ: 目標 {target} pulse", flush=True)
        self.write(116, 4, target)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if self.read(70, 1):
                raise RuntimeError("モータのHardware Errorを検出しました")
            position = self.position()
            if abs(position - target) <= 12:
                time.sleep(0.5)
                return
            time.sleep(0.05)
        raise RuntimeError("目標に到達しませんでした。機械的な端を押し続けず停止します")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--move", action="store_true", help="確認入力後、ID 8だけを小幅に往復")
    parser.add_argument("--delta-deg", type=float, default=10.0, help="現在位置からの角度、1〜15度")
    args = parser.parse_args()
    if not 1 <= args.delta_deg <= 15:
        parser.error("--delta-deg は1〜15度にしてください")

    gripper = Gripper(args.port)
    touched = False
    saved = {}
    try:
        if not gripper.port.openPort():
            raise RuntimeError(f"ポートを開けません: {args.port}")
        if not gripper.port.setBaudRate(57600):
            raise RuntimeError("57600 baudへの設定に失敗しました")
        model = gripper.read(0, 2)
        mode = gripper.read(11, 1)
        torque = gripper.read(64, 1)
        position = gripper.position()
        error = gripper.read(70, 1)
        print(f"ポート={args.port}, ID=8, model={model}, mode={mode}, "
              f"torque={torque}, position={position}, hardware_error={error}", flush=True)
        if not args.move:
            print("読取りのみ完了。モータへの書込みはありません。")
            return

        # XM430-W350's control table is checked against the bundled hardware doc.
        if model != 1020:
            raise RuntimeError("想定のXM430-W350 (model 1020)ではありません。動かしません")
        if mode not in (3, 5):
            raise RuntimeError("位置制御モード3/5ではありません。モードは変更せず中止します")
        if torque or error:
            raise RuntimeError("トルクONまたはHardware Errorです。動かさず中止します")
        if gripper.read(10, 1) & 4:
            raise RuntimeError("時間ベースProfileです。設定を変更せず中止します")
        minimum, maximum = gripper.read(52, 4), gripper.read(48, 4)
        if not 0 <= minimum <= position <= maximum <= 4095:
            raise RuntimeError("現在位置/可動範囲が単回転の想定外です。動かしません")
        delta = round(args.delta_deg * 4096 / 360)
        targets = [max(minimum, min(maximum, position + delta)),
                   max(minimum, min(maximum, position - delta)), position]
        print(f"現在位置を基準に最大±{args.delta_deg}度だけ往復し、ID 8をトルクOFFにします。")
        print("腕・ベースにはアクセスしません。実際の左右は動いたグリッパーを見て判定してください。")
        if input("周囲を空けて非常停止を準備し、実行する場合だけ MOVE8 と入力: ").strip() != "MOVE8":
            print("中止。書込みはありません。")
            return

        # RAM settings only. Never modify mode, offsets, EEPROM or other IDs.
        settings = {100: (2, 80), 108: (4, 10), 112: (4, 10)}
        if mode == 5:
            settings[102] = (2, 37)  # approximately 100 mA, additionally PWM-capped
        saved = {addr: (size, gripper.read(addr, size)) for addr, (size, _) in settings.items()}
        touched = True  # any later failure must attempt torque OFF, including enable ACK loss
        for addr, (size, value) in settings.items():
            gripper.write(addr, size, value)
        gripper.write(116, 4, position)  # synchronize target BEFORE enabling
        gripper.write(64, 1, 1)
        if abs(gripper.position() - position) > 32:
            raise RuntimeError("トルクON時に位置が変化しました。往復を中止します")
        for target in targets:
            gripper.move_to(target)
        print("小幅往復完了。どちらのグリッパーが動いたか確認してください。")
    finally:
        if touched:
            try:
                gripper.write(64, 1, 0)
                if gripper.read(64, 1) != 0:
                    raise RuntimeError("トルクOFFを確認できません")
                print("ID 8のトルクOFFを確認しました。", flush=True)
            except Exception as exc:
                print(f"停止確認失敗: {exc}。非常停止/電源OFFをしてください。", flush=True)
            else:
                for addr, (size, value) in saved.items():
                    try:
                        gripper.write(addr, size, value)
                    except Exception as exc:
                        print(f"RAM設定 {addr} の復元失敗: {exc}", flush=True)
        gripper.port.closePort()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("中断しました。")
        raise SystemExit(130)
    except Exception as exc:
        print(f"中止: {exc}")
        raise SystemExit(1)
