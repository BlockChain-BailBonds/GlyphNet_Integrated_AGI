# Codex — bash

### α

```bash
for {0} in $(seq 0 {1}); do
  {2}
done
```

### γ

```bash
set -euo pipefail
```

### ε

```bash
echo {0}
```

### λ

```bash
{0}(){{ {1} }}
```

### □

```bash
{0}
```

### Ωsh_case

```bash
case {0} in
  {1}) {2} ;;
  *) {3} ;;
esac
```

### Ωsh_if

```bash
if [ {0} ]; then
  {1}
fi
```

### Ωsh_if_elif_else

```bash
if [ {0} ]; then
  {1}
elif [ {2} ]; then
  {3}
else
  {4}
fi
```

### Ωsh_pipe_grep

```bash
{0} | grep -E '{1}'
```

### Ωsh_while_read

```bash
while read -r {0}; do
  {1}
done
```

### Ωcli_echo

```bash
#!/usr/bin/env bash
echo "$@"

```

### Ωsum_numbers

```bash
#!/usr/bin/env bash
s=0; for x in "$@"; do s=$((s + x)); done; echo "$s"

```

### Ωfile_cat

```bash
cat "{0}" 
```

### Ωtimer_tick_5

```bash
while true; do echo "tick"; sleep 5; done

```

### Ωsh_find_grep

```bash
find {0} -type f -print0 | xargs -0 grep -nH --color=always -E {1}
```

### Ω3_stdin_upper_stdout

```bash
tr '[:lower:]' '[:upper:]'
```

### Ω3_file_grep_count

```bash
grep -E {1} {0} | wc -l
```

### Ω3_dir_glob_hash

```bash
python3 - <<'PY'
import glob,hashlib,sys
pat={0}
h=hashlib.sha256()
for p in sorted(glob.glob(pat, recursive=True)):
    try:
        h.update(open(p,'rb').read())
    except: pass
print(h.hexdigest())
PY
```

### Ω3_topk_words

```bash
tr -cs "[:alnum:]_'" "\n" < {0} | tr '[:upper:]' '[:lower:]' | sort | uniq -c | sort -nr | head -n {1}
```

### Ω3_log_extract_errors

```bash
grep 'ERROR' {0} | sort | uniq
```

### Ω3_tar_gzip_dir

```bash
tar -czf {1} -C "$(dirname {0})" "$(basename {0})" && echo {1}
```

### Ω3_stdin_unique_count

```bash
sort | uniq | wc -l
```

### Ωenv_get_print

```bash
echo "${{{0}:-{1}}}" 
```

### Ωenv_set_current

```bash
export {0}={1}
```

### Ωenv_unset

```bash
unset {0}
```

### Ωenv_list

```bash
env
```

### Ωenv_cwd_print

```bash
pwd
```

### Ωenv_chdir

```bash
cd {0}
```

### Ωenv_path_prepend

```bash
export PATH="{0}:$PATH" 
```

### Ωenv_guard_required

```bash
: "${{{0}?Missing required env: {0}}}" 
```

### Ωenv_load_dotenv_min

```bash
set -a; [ -f {0} ] && . {0}; set +a
```

### Ωenv_export_file

```bash
env | sort > {0}
```

### Ωlnx_detect_distro

```bash
source /etc/os-release 2>/dev/null || true; echo "${ID:-unknown}" 
```

### Ωlnx_cc_build

```bash
gcc -O2 -Wall -Wextra -o {1} {0} 
```

### Ωlnx_cc_build_static

```bash
gcc -O2 -static -s -o {1} {0} 
```

### Ωlnx_cpp_build

```bash
g++ -O2 -Wall -Wextra -std=c++17 -o {1} {0} 
```

### Ωlnx_rust_build

```bash
rustc -C opt-level=3 -o {1} {0} 
```

### Ωlnx_go_build

```bash
CGO_ENABLED=0 go build -ldflags '-s -w' -o {1} {0} 
```

### Ωlnx_java_jar

```bash
mkdir -p out && javac -d out {0} && jar --create --file {1} -C out . 
```

### Ωlnx_node_build

```bash
set -e; npm ci; npm run build 
```

### Ωlnx_py_zipapp

```bash
python3 -m zipapp {0} -o {1} -m {2} 
```

### Ωlnx_strip_binary

```bash
strip {0} || true 
```

### Ωlnx_elf_check

```bash
file {0}; echo "---"; (ldd {0} || echo "static or not a dynamic ELF") 
```

### Ωlnx_pkg_tar_gz

```bash
tar -czf {1} -C "$(dirname {0})" "$(basename {0})" 
```

### Ωlnx_pkg_deb_min

```bash
set -euo pipefail
PKG={0}; VER={1}; ARCH={2}; BIN={3}; MAINT={4}; DESC={5}
ROOT="$(mktemp -d)"
install -Dm755 "$BIN" "$ROOT/usr/bin/$(basename "$BIN")"
mkdir -p "$ROOT/DEBIAN"
cat > "$ROOT/DEBIAN/control" <<EOF
Package: $PKG
Version: $VER
Section: utils
Priority: optional
Architecture: $ARCH
Maintainer: $MAINT
Description: $DESC
EOF
dpkg-deb --build "$ROOT" "${PKG}_${VER}_${ARCH}.deb"
echo "${PKG}_${VER}_${ARCH}.deb"

```

### Ωlnx_systemd_unit_install

```bash
set -e
NAME={0}; EXE={1}; USER={2}; OUT=/etc/systemd/system/${{NAME}}.service
sudo tee "$OUT" >/dev/null <<UNIT
[Unit]
Description=$NAME
After=network.target

[Service]
ExecStart=$EXE
Restart=on-failure
User=$USER
WorkingDirectory=/
Environment=LOG_LEVEL=info

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now "$NAME"
systemctl status "$NAME" --no-pager -l || true

```

### Ωlnx_container_build_oci

```bash
docker build -t {0}:{1} {2} 
```

### Ωlnx_container_push

```bash
docker push {0}:{1} 
```

### Ωlnx_build_by_ext

```bash
set -e
SRC={0}; OUT={1}
case "$SRC" in
  *.c)    gcc -O2 -Wall -Wextra -o "$OUT" "$SRC" ;;
  *.cc|*.cpp|*.cxx) g++ -O2 -Wall -Wextra -std=c++17 -o "$OUT" "$SRC" ;;
  *.rs)   rustc -C opt-level=3 -o "$OUT" "$SRC" ;;
  *.go)   CGO_ENABLED=0 go build -ldflags '-s -w' -o "$OUT" "$SRC" ;;
  *.java) mkdir -p out && javac -d out "$SRC" && jar --create --file "$OUT" -C out . ;;
  *) echo "unsupported extension: $SRC" >&2; exit 2 ;;
esac
echo "$OUT"

```

### Ωlnx_appimage_bundle_min

```bash
set -e
APPDIR="$(pwd)/AppDir"; APPNAME={0}; BIN={1}; OUT={2}
rm -rf "$APPDIR"; mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications"
install -m755 "$BIN" "$APPDIR/usr/bin/$APPNAME"
cat > "$APPDIR/$APPNAME.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=$APPNAME
Exec=$APPNAME
Icon=$APPNAME
Categories=Utility;
DESK
mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"
# Provide a 1x1 transparent placeholder icon if none supplied (still concrete)
printf "\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x02\x00\x01\x0b\xe7\x0b\x9d\x00\x00\x00\x00IEND\xaeB`\x82" > "$APPDIR/usr/share/icons/hicolor/256x256/apps/$APPNAME.png"
appimagetool "$APPDIR" "$OUT"
echo "$OUT"

```

### Ωlnx_pkg_rpm_min

```bash
set -euo pipefail
PKG={0}; VER={1}; REL={2}; ARCH={3}; BIN={4}
TOP="$(mktemp -d)"
mkdir -p "$TOP"/{BUILD,RPMS,SOURCES,SPECS,SRPMS}
BNAME="$(basename "$BIN")"
install -Dm755 "$BIN" "$TOP/SOURCES/$BNAME"
SPEC="$TOP/SPECS/${PKG}.spec"
cat > "$SPEC" <<EOF
Name:           $PKG
Version:        $VER
Release:        $REL%{{?dist}}
Summary:        $PKG
License:        MIT
BuildArch:      $ARCH
%description
$PKG
%install
install -Dpm 0755 %{_sourcedir}/$BNAME %{buildroot}%{{_bindir}}/$BNAME
%files
%{{_bindir}}/$BNAME
EOF
rpmbuild --define "_topdir $TOP" -bb "$SPEC"
find "$TOP/RPMS" -type f -name "*.rpm" -print -quit

```

### Ωlinux_bootstrap_all

```bash
set -euo pipefail
KURL={{0}}
BURL={{1}}
WORK={{2}}
ISO_OUT=${{{{4:-}}}}
# Backward-compatible param handling: if {{3}} provided, use it; else default in WORK/out/linux-minimal.iso
if [ -n "{{3}}" ]; then ISO_OUT={{3}}; fi

mkdir -p "$WORK"/{{src,kernel,busybox,rootfs,iso/boot/grub,out}}
cd "$WORK"

# Detect package manager and install deps (best effort; requires sudo)
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends         build-essential bc bison flex libelf-dev libssl-dev cpio xz-utils         wget tar xorriso grub-pc-bin grub-efi-amd64-bin mtools
elif command -v dnf >/dev/null 2>&1; then
  sudo dnf install -y @development-tools bc bison flex elfutils-libelf-devel openssl-devel         cpio xz wget tar xorriso grub2-pc grub2-efi-x64 mtools
elif command -v pacman >/dev/null 2>&1; then
  sudo pacman -Sy --noconfirm base-devel bc bison flex libelf openssl cpio xz wget tar xorriso grub mtools
else
  echo "WARN: Could not detect a supported package manager. Ensure build deps are installed." >&2
fi

# --- Fetch sources ---
cd "$WORK/src"
KFILE="$(basename "$KURL")"
BFILE="$(basename "$BURL")"
[ -f "$KFILE" ] || wget -nv "$KURL"
[ -f "$BFILE" ] || wget -nv "$BURL"

# --- Unpack ---
cd "$WORK/kernel"; tar -xf "$WORK/src/$KFILE" --strip-components=1
cd "$WORK/busybox"; tar -xf "$WORK/src/$BFILE" --strip-components=1

# --- Build kernel (x86_64, bzImage) ---
cd "$WORK/kernel"
make -s ARCH=x86_64 x86_64_defconfig
# A couple of options that help booting with initramfs
./scripts/config --enable CONFIG_BLK_DEV_INITRD || true
./scripts/config --enable CONFIG_DEVTMPFS || true
./scripts/config --enable CONFIG_DEVTMPFS_MOUNT || true
make -s -j"$(nproc)" ARCH=x86_64 bzImage
cp -f arch/x86/boot/bzImage "$WORK/out/vmlinuz"

# --- Build busybox (static) ---
cd "$WORK/busybox"
make -s defconfig
# enable static
sed -i 's/# CONFIG_STATIC is not set/CONFIG_STATIC=y/' .config
make -s -j"$(nproc)"
make -s install CONFIG_PREFIX="$WORK/rootfs"

# --- Rootfs layout ---
cd "$WORK/rootfs"
mkdir -p proc sys dev etc mnt tmp var/run
chmod 1777 tmp
# Create init (PID 1)
cat > init <<'INIT'
#!/bin/sh
mount -t proc none /proc
mount -t sysfs none /sys
mount -t devtmpfs devtmpfs /dev 2>/dev/null || true
echo "Boot OK"
echo "Spawning shell on ttyS0 and tty0"
setsid /bin/sh -c 'exec /bin/sh </dev/ttyS0 >/dev/ttyS0 2>&1' &
exec /bin/sh
INIT
chmod +x init

# --- Create initramfs ---
cd "$WORK/rootfs"
find . -print0 | cpio --null -ov --format=newc | gzip -9 > "$WORK/out/initrd.img"

# --- GRUB ISO ---
cd "$WORK"
cp -f "$WORK/out/vmlinuz" "$WORK/iso/boot/vmlinuz"
cp -f "$WORK/out/initrd.img" "$WORK/iso/boot/initrd.img"
cat > "$WORK/iso/boot/grub/grub.cfg" <<'GRUBCFG'
set timeout=1
set default=0

menuentry 'Minimal Linux (serial+console)' {{{{
    linux /boot/vmlinuz console=ttyS0 console=tty0
    initrd /boot/initrd.img
}}}}

GRUBCFG

ISO_TMP="$WORK/out/linux-minimal.iso"
if command -v grub-mkrescue >/dev/null 2>&1; then
  grub-mkrescue -o "$ISO_TMP" "$WORK/iso" 2>/dev/null || grub-mkrescue -o "$ISO_TMP" "$WORK/iso"
else
  echo "ERROR: grub-mkrescue not found (need grub + xorriso)" >&2
  exit 2
fi

# Finalize outputs
mkdir -p "$WORK/out"
[ -z "$ISO_OUT" ] && ISO_OUT="$ISO_TMP" || cp -f "$ISO_TMP" "$ISO_OUT"

# Manifest
{{
  echo "# Minimal Linux boot artifacts"
  echo "KERNEL:   $WORK/out/vmlinuz"
  echo "INITRD:   $WORK/out/initrd.img"
  echo "GRUBCFG:  $WORK/iso/boot/grub/grub.cfg"
  echo "ISO:      $ISO_OUT"
}} > "$WORK/out/manifest.txt"

echo "Artifacts:"
cat "$WORK/out/manifest.txt"

```

### Ωlinux_qemu_iso

```bash
qemu-system-x86_64 -m {{0}} -enable-kvm -cpu host -nographic -serial mon:stdio -cdrom {{1}}

```

### Ωandr_sdk_setup_linux

```bash
set -euo pipefail
ANDROID_HOME={0}
API={2}
BUILDTOOLS={3}
mkdir -p "$ANDROID_HOME"
cd "$ANDROID_HOME"
if [ ! -d "cmdline-tools" ]; then
  curl -L {1} -o cmdline-tools.zip
  mkdir -p cmdline-tools
  unzip -q cmdline-tools.zip -d cmdline-tools_tmp
  # move into cmdline-tools/latest as expected
  mkdir -p cmdline-tools/latest
  mv cmdline-tools_tmp/cmdline-tools/* cmdline-tools/latest/ || mv cmdline-tools_tmp/* cmdline-tools/latest/ || true
  rm -rf cmdline-tools_tmp cmdline-tools.zip
fi
export ANDROID_SDK_ROOT="$ANDROID_HOME"
export PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator:$PATH"
yes | sdkmanager --licenses >/dev/null
sdkmanager "platform-tools" "platforms;android-${{API}}" "build-tools;${{BUILDTOOLS}}" "emulator" "system-images;android-${{API}};google_apis;x86_64"

```

### Ωandr_avd_create_start

```bash
set -e
AVD={0}
API={1}
avdmanager create avd -n "$AVD" -k "system-images;android-${{API}};google_apis;x86_64" --device "pixel" --force || true
nohup emulator -avd "$AVD" -no-snapshot -no-window -no-audio -gpu swiftshader_indirect >/tmp/emulator.log 2>&1 &
adb wait-for-device
# wait until boot completed
until adb shell getprop sys.boot_completed 2>/dev/null | grep -q "1"; do sleep 1; done
adb shell input keyevent 82 || true

```

### Ωandr_gradle_wrapper

```bash
gradle wrapper --gradle-version {0} 
```

### Ωandr_gradle_build_apk_debug

```bash
./gradlew assembleDebug 
```

### Ωandr_gradle_build_release_aab

```bash
./gradlew bundleRelease 
```

### Ωandr_keystore_create

```bash
keytool -genkeypair -v -keystore {0} -alias {1} -keyalg RSA -keysize 4096 -validity 36500       -storepass {2} -keypass {2} -dname {3}

```

### Ωandr_zipalign

```bash
zipalign -v -p 4 {0} {1} 
```

### Ωandr_apksigner

```bash
apksigner sign --ks {0} --ks-pass pass:{1} --out {3} {2}
```

### Ωandr_install_apk

```bash
adb install -r {0} 
```

### Ωandr_uninstall_pkg

```bash
adb uninstall {0} 
```

### Ωandr_adb_shell

```bash
adb shell {0} 
```

### Ωandr_logcat_grep

```bash
adb logcat | grep -E {0} 
```

### Ωandr_instrumentation_test

```bash
./gradlew connectedAndroidTest 
```

### Ωandr_bundle_install

```bash
java -jar {0} build-apks --bundle={1} --output={2} --connected-device --overwrite
java -jar {0} install-apks --apks={2}

```

### Ωandr_project_scaffold_min

```bash
set -euo pipefail
APPID={0}
APPNAME={1}
DIR={2}
AGP={3}
GRADLE_VER={4}
SDK={5}
API={6}
MINSDK={7}

mkdir -p "$DIR"
cd "$DIR"
mkdir -p app/src/main/java/$(echo "$APPID" | tr '.' '/') app/src/androidTest/java/$(echo "$APPID" | tr '.' '/') app/src/test/java/$(echo "$APPID" | tr '.' '/')
mkdir -p app/src/main/res/values

# Write settings.gradle, root build.gradle, app/build.gradle, gradle.properties, manifest, MainActivity.kt
cat > settings.gradle <<SET
{rootProject.name = "{0}"
include(":app")
}
SET

cat > build.gradle <<ROOT
{buildscript {{{{
    repositories {{{{
        google()
        mavenCentral()
    }}}}
    dependencies {{{{
        classpath "com.android.tools.build:gradle:{0}"
    }}}}
}}}}
allprojects {{{{
    repositories {{{{
        google()
        mavenCentral()
    }}}}
}}}}
}
ROOT

cat > gradle.properties <<PROPS
{org.gradle.jvmargs=-Xmx2g -Dfile.encoding=UTF-8
android.useAndroidX=true
android.enableJetifier=true
}
PROPS

cat > app/build.gradle <<APP
{apply plugin: "com.android.application"

android {{{{
    namespace "{0}"
    compileSdk {1}

    defaultConfig {{{{
        applicationId "{0}"
        minSdk {2}
        targetSdk {1}
        versionCode 1
        versionName "1.0"
        testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"
    }}}}

    buildTypes {{{{
        release {{{{
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }}}}
    }}}}
}}}}

dependencies {{{{
    implementation "androidx.appcompat:appcompat:1.7.0"
    implementation "com.google.android.material:material:1.12.0"
    testImplementation "junit:junit:4.13.2"
    androidTestImplementation "androidx.test.ext:junit:1.2.1"
    androidTestImplementation "androidx.test.espresso:espresso-core:3.6.1"
}}}}
}
APP

cat > app/src/main/AndroidManifest.xml <<MANI
{<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="{0}">
  <application android:label="{1}" android:allowBackup="true" android:supportsRtl="true">
    <activity android:name=".MainActivity">
      <intent-filter>
        <action android:name="android.intent.action.MAIN" />
        <category android:name="android.intent.category.LAUNCHER" />
      </intent-filter>
    </activity>
  </application>
</manifest>
}
MANI

PKGDIR="app/src/main/java/$(echo "$APPID" | tr '.' '/')"
cat > "$PKGDIR/MainActivity.kt" <<KOT
{package {0}

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {{{{
    override fun onCreate(savedInstanceState: Bundle?) {{{{
        super.onCreate(savedInstanceState)
        val tv = TextView(this)
        tv.text = "Hello, Android!"
        setContentView(tv)
    }}}}
}}}}
}
KOT

echo "{.gradle/
/.idea/
/local.properties
**/build/
.DS_Store
}" > .gitignore

# Gradle wrapper & SDK config
export ANDROID_SDK_ROOT="$SDK"
export ANDROID_HOME="$SDK"
export PATH="$SDK/cmdline-tools/latest/bin:$SDK/platform-tools:$SDK/emulator:$PATH"

gradle wrapper --gradle-version "$GRADLE_VER"

# Create local.properties pointing to SDK (non-committed)
echo "sdk.dir=$SDK" > local.properties

# Preflight: ensure platform & build tools exist
yes | sdkmanager --licenses >/dev/null || true
sdkmanager "platform-tools" "platforms;android-${{API}}" "build-tools;${{API}}.0.0" || true

echo "Project scaffolded at $DIR"

```

### Ωios_check_xcode

```bash
set -e; xcode-select -p; xcodebuild -version; xcrun --version; xcrun simctl list | head -n 25
```

### Ωios_select_xcode

```bash
sudo xcode-select --switch {0}
```

### Ωios_accept_licenses

```bash
sudo xcodebuild -license accept || true
```

### Ωios_sim_runtimes

```bash
xcrun simctl list runtimes
```

### Ωios_sim_devices

```bash
xcrun simctl list devices
```

### Ωios_sim_create

```bash
xcrun simctl create {0} "{1}" "{2}" 
```

### Ωios_sim_boot

```bash
xcrun simctl boot {0} && xcrun simctl bootstatus {0} -b && open -a Simulator
```

### Ωios_sim_install_launch

```bash
xcrun simctl install {0} {1} && xcrun simctl launch {0} {2}
```

### Ωios_sim_shutdown_all

```bash
xcrun simctl shutdown all || true
```

### Ωios_pods_install

```bash
(bundle exec pod install || pod install)
```

### Ωios_spm_resolve_project

```bash
xcodebuild -resolvePackageDependencies -project {0} -scheme {1}
```

### Ωios_spm_resolve_workspace

```bash
xcodebuild -resolvePackageDependencies -workspace {0} -scheme {1}
```

### Ωios_build_sim_debug

```bash
xcodebuild -scheme {0} -configuration Debug -sdk iphonesimulator -destination 'platform=iOS Simulator,name={1},OS={2}' clean build
```

### Ωios_build_device_release

```bash
xcodebuild -scheme {0} -configuration Release -sdk iphoneos clean build
```

### Ωios_test_sim

```bash
xcodebuild -scheme {0} -configuration Debug -sdk iphonesimulator -destination 'platform=iOS Simulator,name={1},OS={2}' clean test
```

### Ωios_archive

```bash
xcodebuild -scheme {0} -configuration Release -sdk iphoneos -archivePath {1} archive -allowProvisioningUpdates
```

### Ωios_export_ipa

```bash
xcodebuild -exportArchive -archivePath {0} -exportOptionsPlist {1} -exportPath {2} -allowProvisioningUpdates
```

### Ωios_codesign_identities

```bash
/usr/bin/security find-identity -v -p codesigning || true
```

### Ωios_profiles_list

```bash
ls -1 "$HOME/Library/MobileDevice/Provisioning Profiles" || true
```

### Ωios_plist_set

```bash
/usr/libexec/PlistBuddy -c 'Set :{0} {1}' {2}
```

### Ωios_set_bundle_id

```bash
/usr/libexec/PlistBuddy -c 'Set :CFBundleIdentifier {0}' {1}
```

### Ωios_set_display_name

```bash
/usr/libexec/PlistBuddy -c 'Set :CFBundleDisplayName {0}' {1}
```

### Ωios_version_bump

```bash
/usr/libexec/PlistBuddy -c 'Set :CFBundleShortVersionString {0}' {2} && /usr/libexec/PlistBuddy -c 'Set :CFBundleVersion {1}' {2}
```

### Ωios_keychain_create

```bash
security create-keychain -p {1} {0} && security set-keychain-settings -lut 21600 {0} && security unlock-keychain -p {1} {0}
```

### Ωios_keychain_import_p12

```bash
security import {0} -k {1} -P {2} -A && security set-key-partition-list -S apple-tool:,apple: -s -k {3} {1}
```

### Ωios_keychain_use_default

```bash
security list-keychains -d user -s {0} && security default-keychain -s {0}
```

### Ωios_keychain_delete

```bash
security delete-keychain {0} || true
```

### Ωios_ipa_install_device

```bash
ios-deploy --bundle {0} {1}
```

### Ωios_ipa_unzip_to_app

```bash
set -e; OUT={1}; rm -rf "$OUT"; mkdir -p "$OUT"; unzip -q {0} -d "$OUT"; echo "$OUT/Payload" && ls "$OUT/Payload" 
```

### Ωlib_log_json

```bash
log_json {0} {1}  # LEVEL MSG [k=v ...]
```

### Ωlib_base_bash

```bash
# ---- Ωlib_base_bash: bash 4+ helpers ----
# JSON log (minimal; no jq). Usage: log_json LEVEL MSG [k=v ...]
log_json(){{ 
  local lvl="$1"; shift; local msg="$1"; shift
  local ts; ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  local rest=""; for kv in "$@"; do rest="$rest, \"${{kv%%=*}}\": \"${{kv#*=}}\""; done
  >&2 printf '{{ "ts":"%s", "level":"%s", "msg":"%s"%s }}\n' "$ts" "$lvl" "$msg" "$rest"
}}

# CLI parse: --k=v | --k v | -k v ; emits associative array ARGS and array POSITIONALS
cli_parse(){{
  declare -gA ARGS; declare -g POSITIONALS; POSITIONALS=()
  local a; while (( "$#" )); do
    a="$1"; shift
    if [[ "$a" == --*=* ]]; then ARGS["${{a%%=*#}}"]="${{a#*=}}"; continue; fi
    if [[ "$a" == --* ]]; then 
      local k="${{a#--}}"; if [[ "$1" != -* && -n "$1" ]]; then ARGS["$k"]="$1"; shift; else ARGS["$k"]="true"; fi; continue
    fi
    if [[ "$a" == -* && "${{#a}}" -gt 1 ]]; then
      local k="${{a:1:1}}"; if [[ "$1" != -* && -n "$1" ]]; then ARGS["$k"]="$1"; shift; else ARGS["$k"]="true"; fi; continue
    fi
    POSITIONALS+=("$a")
  done
}}

# FS helpers
read_text(){{ local p="$1"; cat "$p"; }}
write_text(){{ local p="$1"; shift; mkdir -p "$(dirname "$p")"; printf "%s" "$*" > "$p"; }}
mkdir_p(){{ mkdir -p "$1"; }}
exists(){{ [[ -e "$1" ]]; }}

# SHA256 (requires sha256sum or shasum -a 256)
sha256_file(){{
  local p="$1"
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$p" | awk '{{print $1}}'; 
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$p" | awk '{{print $1}}'; 
  else echo "no sha256 program" >&2; return 1; fi
}}

# HTTP GET (curl or wget)
http_get(){{
  local url="$1"
  if command -v curl >/dev/null 2>&1; then curl -fsSL "$url";
  elif command -v wget >/dev/null 2>&1; then wget -qO- "$url";
  else echo "no curl/wget" >&2; return 1; fi
}}

# Retry: retry N DELAY cmd...
retry(){{ local n="$1"; local delay="$2"; shift 2; local i=1; while true; do "$@" && return 0; (( i>=n )) && return 1; sleep "$delay"; ((delay*=2)); ((i++)); done }}

```

### Ωcx_build_ext_pack_release

```bash
set -euo pipefail
SRC={0}; OUTBIN={1}; OUTTGZ={2}; OUTMAN={3}
case "$SRC" in
  *.c)    gcc -O2 -Wall -Wextra -o "$OUTBIN" "$SRC" ;;
  *.cc|*.cpp|*.cxx) g++ -O2 -Wall -Wextra -std=c++17 -o "$OUTBIN" "$SRC" ;;
  *.rs)   rustc -C opt-level=3 -o "$OUTBIN" "$SRC" ;;
  *.go)   CGO_ENABLED=0 go build -ldflags '-s -w' -o "$OUTBIN" "$SRC" ;;
  *) echo "unsupported extension: $SRC" >&2; exit 2 ;;
esac
strip "$OUTBIN" || true
FILEINFO=$(file "$OUTBIN" || true)
LDDINFO=$(ldd "$OUTBIN" 2>&1 || echo "static or not a dynamic ELF")
mkdir -p "$(dirname "$OUTTGZ")"
tar -czf "$OUTTGZ" -C "$(dirname "$OUTBIN")" "$(basename "$OUTBIN")"
if command -v sha256sum >/dev/null 2>&1; then SHASUM=$(sha256sum "$OUTTGZ" | awk '{{print $1}}'); 
else SHASUM=$(shasum -a 256 "$OUTTGZ" | awk '{{print $1}}'); fi
printf '{{\n  "src":"%s",\n  "bin":"%s",\n  "tgz":"%s",\n  "sha256":"%s",\n  "file":"%s",\n  "ldd":"%s"\n}}\n'       "$SRC" "$OUTBIN" "$OUTTGZ" "$SHASUM" "$FILEINFO" "$LDDINFO" > "$OUTMAN"
echo "$OUTMAN"

```

### Ωcx_parallel_stdin_map_cmd

```bash
# Usage: echo -e "a\nb\nc" | {{glyph}}.format("4","echo {{}}")
PAR={0}; CMDT={1}
xargs -P "$PAR" -I{{}} sh -c "$CMDT"

```

### Ωcx_log_rotate_sizeN

```bash
FILE={0}; MAXB={1}; KEEP={2}
sz=$(stat -c%s "$FILE" 2>/dev/null || stat -f%z "$FILE" 2>/dev/null || echo 0)
if [ "$sz" -le "$MAXB" ]; then exit 0; fi
for i in $(seq "$KEEP" -1 2); do if [ -f "$FILE.$((i-1))" ]; then mv -f "$FILE.$((i-1))" "$FILE.$i"; fi; done
if [ -f "$FILE" ]; then mv -f "$FILE" "$FILE.1"; : > "$FILE"; fi

```

### Ωcx_backup_dir_timestamped

```bash
SRC={0}; PREFIX={1}; OUTDIR={2}
TS=$(date -u +"%Y%m%dT%H%M%SZ")
mkdir -p "$OUTDIR"
OUT="$OUTDIR/${{PREFIX}}_${{TS}}.tgz"
tar -czf "$OUT" -C "$(dirname "$SRC")" "$(basename "$SRC")"
echo "$OUT"

```
