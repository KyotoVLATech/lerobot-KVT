
import asyncio
from src.lerobot.robots.iloha.iloha_controller.robstride.src.robstride import RobStrideController, RobStride

async def test_feedback():
    port = "/dev/ttyUSB2"
    motor_id = 1
    motor = RobStride(id=motor_id, offset=0.0)
    controller = RobStrideController(port=port, motors=[motor])
    
    await controller.connect()
    print(f"\n--- Feedback Test (Motor {motor_id} on {port}) ---")
    
    feedback = await controller.get_motor_feedback(motor_id)
    if feedback:
        print(f"Feedback Success!")
        print(f"  Temperature: {feedback['temp']} ℃")
        print(f"  Torque Raw:  {feedback['torque']}")
        print(f"  Velocity:    {feedback['velocity']}")
    else:
        print("Feedback failed.")
        
    await controller.disconnect()

if __name__ == "__main__":
    asyncio.run(test_feedback())
