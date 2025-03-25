## Android Development
This page contains notes related to Whisper.cpp and the Android examples.

### Installation
Download the command line tools from the
[Android developer website](https://developer.android.com/studio/index.html#command-line-tools-only).
Unzip them and the move them into:
```console
$ mkdir -p ~/Android/Sdk/cmdline-tools/latest

```
Then set the following environment variables:
```console
export ANDROID_HOME=~/Android/Sdk
export PATH=$ANDROID_HOME/cmdline-tools/latest/bin:$PATH
export PATH=$ANDROID_HOME/platform-tools:$PATH
```

I'm using java 21.0.6 and needed to upgrade gradle:
```console
$ cd ~/Downloads
$ wget https://services.gradle.org/distributions/gradle-8.5-bin.zip
$ unzip gradle-8.5-bin.zip
$ export PATH=$PATH:~/Downloads/gradle-8.5/bin
```

Install necessary Android SDK components:
```console
$ sdkmanager "build-tools;34.0.0" "platform-tools" "platforms;android-34"
```

With that done, I was able to at least clean the `examples/whisper.android.java`
```console
$ gradle clean
```

```console
$ gradle assembleDebug
```

### Running the Android Emulator

Create the AVD (Android Virtual Device) with the following script:
```console
$ cat create-avd.sh 
JAVA21_HOME="/usr/lib/jvm/java-21-openjdk-amd64"
ANDROID_SDK="$HOME/Android/Sdk"

export JAVA_HOME="$JAVA21_HOME"
export PATH="$JAVA_HOME/bin:$PATH"

echo "JAVA_HOME=$JAVA21_HOME"
echo "ANDROID_SDK=$ANDROID_SDK"

#"$ANDROID_SDK/cmdline-tools/latest/bin/sdkmanager" list | grep system-images

#sdkmanager "system-images;android-30;google_apis;x86_64"

avdmanager create avd --name "whisper_test_device" --package "system-images;android-30;google_apis;x86_64" --device "pixel"
```
Start the AVD with the following script:
```console
$ cat start-avd.sh 
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

$ANDROID_HOME/emulator/emulator -avd whisper_test_device -no-audio -no-window -no-boot-anim &
```

Install the app on the AVD:
```console
#!/bin/bash

# Set Java 11 for building
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

echo "Building and installing app..."
./gradlew installDebug

echo "Launching app..."
$ANDROID_HOME/platform-tools/adb shell am start -n "com.litongjava.whisper.android.java/.MainActivity"

echo "Showing logs (press Ctrl+C to stop):"
$ANDROID_HOME/platform-tools/adb logcat | grep -i whisper
```

For the whisper.android.java app we need to copy a model into the directory
`app/src/main/assets/models`:
```console
$ cp ../../models/ggml-base.en.bin app/src/main/assets/models/
```
We also have to copy a sample audio file into the directory:
```console
$ mkdir app/src/main/assets/samples
$ cp ../../models/ggml-tiny.en.bin app/src/main/assets/models/ggml-tiny.bin

```


```console
$ adb shell cmd package resolve-activity --brief com.litongjava.whisper.android.java
priority=0 preferredOrder=0 match=0x108000 specificIndex=-1 isDefault=false
com.litongjava.whisper.android.java/.MainActivity
```

### Debugging Native Code
The most basic way is to add log statements to the C++ code and then view the
```c++
#include <android/log.h>

__android_log_print(ANDROID_LOG_INFO, "danbev", "whisper_state_init................");
```
And this will visable in adb logcat:
```console
03-25 11:52:40.193  7293  7348 I danbev  : whisper_init_state.................
```
