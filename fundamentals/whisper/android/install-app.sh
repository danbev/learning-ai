#!/bin/bash

set -e

# Set Java 11 for building
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

echo "Building and installing app..."
./gradlew clean installDebug

echo "Launching app..."
$ANDROID_HOME/platform-tools/adb shell am start -n "com.litongjava.whisper.android.java/.MainActivity"

echo "Showing logs (press Ctrl+C to stop):"
$ANDROID_HOME/platform-tools/adb logcat | grep -i whisper
