JAVA21_HOME="/usr/lib/jvm/java-21-openjdk-amd64"
ANDROID_SDK="$HOME/Android/Sdk"

export JAVA_HOME="$JAVA21_HOME"
export PATH="$JAVA_HOME/bin:$PATH"

echo "JAVA_HOME=$JAVA21_HOME"
echo "ANDROID_SDK=$ANDROID_SDK"

#"$ANDROID_SDK/cmdline-tools/latest/bin/sdkmanager" list | grep system-images

#sdkmanager "system-images;android-30;google_apis;x86_64"

avdmanager create avd --name "whisper_test_device" --package "system-images;android-30;google_apis;x86_64" --device "pixel"
