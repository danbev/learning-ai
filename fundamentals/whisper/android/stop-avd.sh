#!/bin/bash

JAVA21_HOME="/usr/lib/jvm/java-21-openjdk-amd64"
ANDROID_SDK="$HOME/Android/Sdk"

export JAVA_HOME="$JAVA21_HOME"
export PATH="$JAVA_HOME/bin:$PATH"

echo "JAVA_HOME=$JAVA21_HOME"
echo "ANDROID_SDK=$ANDROID_SDK"

# Send shutdown command to the emulator
#$ANDROID_HOME/platform-tools/adb -s emulator-5554 emu kill

echo "Stopping any running emulators..."
if command -v adb &> /dev/null; then
    adb devices | grep emulator | cut -f1 | while read line; do
        echo "Stopping $line"
        adb -s $line emu kill
    done
else
    echo "ADB not found, using pkill instead"
    pkill -f "emulator-.*"
fi

echo "Waiting for emulator to fully stop..."
sleep 3

echo "Done. All emulators stopped."
