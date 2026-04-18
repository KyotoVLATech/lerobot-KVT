#!/bin/bash

# check for root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (use sudo)"
  exit 1
fi

RULES_FILE="/etc/udev/rules.d/99-iloha-usb-serial.rules"

echo "=========================================="
echo "   Iloha Robot USB Persistence Setup"
echo "=========================================="
echo "1) Apply Rules (設定する)"
echo "2) Remove Rules (解除する)"
echo "3) Exit (終了)"
echo "------------------------------------------"
read -p "Select an option [1-3]: " Choice

case $Choice in
  1)
    echo "Applying udev rules..."
    cat <<EOF > "$RULES_FILE"
# Iloha Dynamixel Persistence Rules
# Left Dynamixel (Serial: FTAK89LL)
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6014", ATTRS{serial}=="FTAK89LL", SYMLINK+="ttyUSB_LeftDynamixel", MODE="0666"

# Right Dynamixel (Serial: FT89FK7J)
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6014", ATTRS{serial}=="FT89FK7J", SYMLINK+="ttyUSB_RightDynamixel", MODE="0666"
EOF
    echo "Rules created at $RULES_FILE"
    ;;

  2)
    if [ -f "$RULES_FILE" ]; then
      echo "Removing rules file: $RULES_FILE"
      rm "$RULES_FILE"
      echo "Rules removed."
    else
      echo "Rules file does not exist."
    fi
    ;;

  *)
    echo "Exiting."
    exit 0
    ;;
esac

# reload udev
echo "Reloading udev rules..."
udevadm control --reload-rules
udevadm trigger

if [ "$Choice" == "1" ]; then
  echo "✅ Persistent symlinks created/updated:"
  ls -l /dev/ttyUSB_* 2>/dev/null || echo "No symlinks found yet. Please unplug and replug the devices if they don't appear."
  echo ""
  echo "Now you can use /dev/ttyUSB_LeftDynamixel and /dev/ttyUSB_RightDynamixel in your config."
elif [ "$Choice" == "2" ]; then
  echo "✅ Symlinks should be removed. (Persistent nodes like /dev/ttyUSBn will remain)"
fi
