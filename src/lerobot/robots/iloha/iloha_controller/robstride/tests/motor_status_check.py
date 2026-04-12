import asyncio
import sys
import argparse
import serial_asyncio
from lerobot.robots.iloha.iloha_controller.robstride.src.robstride import RobStride, RobStrideController
from lerobot.robots.iloha.iloha_controller.robstride.src.constants import ParameterIndex, RunMode

async def main(port, scan_range):
    # ダミーのモーターリストでコントローラー作成
    dummy_motor = RobStride(id=1, offset=0.0)
    controller = RobStrideController(port=port, motors=[dummy_motor])
    
    print(f"📡 {port} に直接接続を開始します...")
    try:
        try:
            coro = serial_asyncio.open_serial_connection(url=port, baudrate=controller.baudrate)
            controller.reader, controller.writer = await coro
            print(f"✅ シリアルポート {port} をオープンしました。")
        except Exception as e:
            print(f"❌ {port} を開けませんでした: {e}")
            return
            
        print("\n" + "═"*75)
        print(f"📊 RobStride モータースキャン開始 (ID 1 ～ {scan_range}): {port}")
        print("═"*75)
        
        real_present_motors = []
        for motor_id in range(1, scan_range + 1):
            print(f"  ID {motor_id:2d} を確認中...", end="\r")
            val = await controller.get_parameter(motor_id, ParameterIndex.RUN_MODE, "uint8")
            if val is not None:
                real_present_motors.append(motor_id)
                print(f"  ID {motor_id:2d} [FOUND]          ")

        if not real_present_motors:
            print("\n📭 モーターが見つかりませんでした。")
            return

        print(f"\n合計 {len(real_present_motors)} 台のモーターを発見しました。詳細データを取得中...\n")

        for motor_id in real_present_motors:
            if motor_id not in controller.motors:
                controller.motors[motor_id] = RobStride(id=motor_id, offset=0.0)
                
            status = await controller.get_motor_status_comprehensive(motor_id)
            versions = await controller.get_version(motor_id)
            if status:
                hw_str = versions.get('hw', 'Unknown') if versions else 'Unknown'
                sw_str = versions.get('sw', 'Unknown') if versions else 'Unknown'
                
                print(f"╔═════════════════════════════════════════════════════════════════════╗")
                print(f"║ [ MOTOR ID: {motor_id:2d} ]  HW: {hw_str:10} / SW: {sw_str:10}           ║")
                print(f"╠═════════════════════════════════════════════════════════════════════╣")
                
                # --- セクション1: 基本状態とエラー ---
                print(f"║ ■ 基本状態・異常検知                                           ║")
                print(f"║   モード:         {status.get('run_mode', 'Unknown'):<42}║")
                faults = status.get('fault_list', [])
                fault_str = ", ".join(faults)
                if not faults or "None" in fault_str:
                    print(f"║   ステータス:      ✅ 正常                                      ║")
                else:
                    print(f"║   異常検知:       ❌ {fault_str[:42]:<42}║")
                
                # --- セクション2: 内部温度詳細 (追加) ---
                print(f"║                                                                ║")
                print(f"║ ■ 内部温度 (Temperatures)                                      ║")
                m_temp = status.get('motor_temp', 0.0)
                b_temp = status.get('board_temp', 0.0)
                mcu_t = status.get('mcu_temp', 0.0)
                d_temp = status.get('drv_temp', 0.0)
                temp_alert = "⚠️" if m_temp > 60 or b_temp > 60 else "  "
                print(f"║ {temp_alert} モーター:  {m_temp:5.1f} ℃  |  基板:    {b_temp:5.1f} ℃                 ║")
                print(f"║    MCU:       {mcu_t:5.1f} ℃  |  ドライバ: {d_temp:5.1f} ℃                 ║")

                # --- セクション3: リアルタイム計測器 ---
                print(f"║                                                                ║")
                print(f"║ ■ リアルタイム計測 (Real-time Metrics)                         ║")
                print(f"║   現在位置:       {status.get('mech_pos', 0.0):10.4f} rad   | バス電圧:   {status.get('vbus', 0.0):7.2f} V      ║")
                print(f"║   現在速度:       {status.get('mech_velo', 0.0):10.4f} rad/s | フィルタ電流: {status.get('iqf', 0.0):7.4f} A      ║")
                print(f"║   生電流 (Iq):    {status.get('iq_raw', 0.0):10.4f} A     | 無駄電流 (Id): {status.get('id_raw', 0.0):7.4f} A      ║")
                print(f"║   フィードバックトルク: {status.get('torque_fdb', 0.0):7.3f} Nm                               ║")

                # --- セクション4: 目標値と制限 ---
                print(f"║                                                                ║")
                print(f"║ ■ 目標値・制限設定 (Targets & Limits)                          ║")
                print(f"║   目標位置:       {status.get('loc_ref', 0.0):10.4f} rad   | 目標速度:   {status.get('spd_ref', 0.0):7.4f} rad/s ║")
                print(f"║   目標電流(Iqf):   {status.get('iq_ref', 0.0):10.4f} A     | トルク制限: {status.get('limit_torque', 0.0):7.2f} Nm    ║")
                print(f"║   最大速度(PP):    {status.get('vel_max', 0.0):10.2f} rad/s | 速度制限:   {status.get('limit_spd', 0.0):7.2f} rad/s ║")
                print(f"║   最大加速度(PP):  {status.get('acc_set', 0.0):10.2f} rad/s²| 電流制限:   {status.get('limit_cur', 0.0):7.4f} A      ║")

                # --- セクション5: PIDゲイン ---
                print(f"║                                                                ║")
                print(f"║ ■ 制御ゲイン設定 (Control Gains)                               ║")
                print(f"║   位置 P:         {status.get('loc_kp', 0.0):10.2f} / Filter: {status.get('spd_filt_gain', 0.0):5.2f}           ║")
                print(f"║   速度 P / I:      {status.get('spd_kp', 0.0):10.4f} / {status.get('spd_ki', 0.0):10.4f}                ║")
                print(f"║   電流 P / I:      {status.get('cur_kp', 0.0):10.4f} / {status.get('cur_ki', 0.0):10.4f}                ║")
                
                # --- セクション6: システム・通信 ---
                print(f"║                                                                ║")
                print(f"║ ■ システム・通信設定                                           ║")
                print(f"║   CAN Timeout:    {status.get('cantimeout', 0):<10d} | EPScan:     {status.get('epscan_time', 0):<7d} ms    ║")
                print(f"║   追加オフセット:   {status.get('add_offset', 0.0):10.4f} rad                                  ║")
                print(f"╚═════════════════════════════════════════════════════════════════════╝")
                print()
                
        print("═"*75)

    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await controller.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RobStride モーター内部パラメータ網羅確認ツール")
    parser.add_argument("port", nargs="?", default="/dev/ttyUSB2", help="シリアルポート")
    parser.add_argument("--scan", type=int, default=10, help="最大スキャンID (1-10)")
    args = parser.parse_args()
    asyncio.run(main(args.port, args.scan))
