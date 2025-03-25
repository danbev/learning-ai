JAVA21_HOME="/usr/lib/jvm/java-21-openjdk-amd64"
ANDROID_SDK="$HOME/Android/Sdk"

export JAVA_HOME="$JAVA21_HOME"
export PATH="$JAVA_HOME/bin:$PATH"

echo "JAVA_HOME=$JAVA21_HOME"
echo "ANDROID_SDK=$ANDROID_SDK"

# Start ADB server if not running
if ! pgrep -f "adb server" > /dev/null; then
    echo "Starting ADB server..."
    $ANDROID_HOME/platform-tools/adb start-server
fi

# Start emulator
echo "Starting emulator..."
$ANDROID_HOME/emulator/emulator -avd whisper_test_device -no-audio &

# Wait for device to boot
echo "Waiting for device to boot..."
$ANDROID_HOME/platform-tools/adb wait-for-device
$ANDROID_HOME/platform-tools/adb shell 'while [[ -z $(getprop sys.boot_completed) ]]; do sleep 1; done'
echo "Emulator is ready to use"

$ANDROID_HOME/emulator/emulator -avd whisper_test_device -memory 2048 &
