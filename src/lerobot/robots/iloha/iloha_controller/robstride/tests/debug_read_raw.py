
import asyncio
import struct
from src.lerobot.robots.iloha.iloha_controller.robstride.src.robstride import RobStrideController, RobStride
from src.lerobot.robots.iloha.iloha_controller.robstride.src.constants import ParameterIndex

async def debug_raw():
    port = "/dev/ttyUSB2"
    motor_id = 1
    motor = RobStride(id=motor_id, offset=0.0)
    controller = RobStrideController(port=port, motors=[motor])
    
    await controller.connect()
    
    indices = [
        ParameterIndex.MCU_TEMP,
        ParameterIndex.MOTOR_TEMP,
        ParameterIndex.VBUS,
        ParameterIndex.FAULT_STA
    ]
    
    print(f"\n--- Raw Data Debug (Motor {motor_id} on {port}) ---")
    for idx in indices:
        # Use low-level _read_parameter to see all 4 bytes
        raw_bytes = await controller._read_parameter(motor_id, idx.value)
        if raw_bytes:
            hex_str = " ".join([f"{b:02X}" for b in raw_bytes])
            print(f"Index {hex(idx.value)} ({idx.name}): {hex_str}")
            
            # Try different decodings
            try:
                le_u16_low = struct.unpack('<H', raw_bytes[0:2])[0]
                le_u16_high = struct.unpack('<H', raw_bytes[2:4])[0]
                be_u16_low = struct.unpack('>H', raw_bytes[0:2])[0]
                f32 = struct.unpack('<f', raw_bytes)[0]
                print(f"  LE_U16[0:2]: {le_u16_low} | LE_U16[2:4]: {le_u16_high}")
                print(f"  BE_U16[0:2]: {be_u16_low} | Float32: {f32:.4f}")
            except:
                pass
        else:
            print(f"Index {hex(idx.value)} ({idx.name}): FAILED (No response)")
            
    await controller.disconnect()

if __name__ == "__main__":
    asyncio.run(debug_raw())
