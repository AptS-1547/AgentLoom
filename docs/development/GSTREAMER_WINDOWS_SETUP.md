# GStreamer Windows Setup

This repository can use the official GStreamer MSVC prebuilt package for media
pipeline probes and future WebRTC experiments.

Current local package path:

```text
<gstreamer-root>
```

The CMake cache variable is:

```cmake
GSTREAMER_ROOT
```

Build the probe:

```powershell
cmake --build build/x64-Release-Tests --target media_gstreamer_probe --config Release --parallel
```

Run the probe from PowerShell:

```powershell
$env:PATH = "<gstreamer-root>\bin;$env:PATH"
$env:GST_PLUGIN_PATH = "<gstreamer-root>\lib\gstreamer-1.0"
build\x64-Release-Tests\Release\media_gstreamer_probe.exe
```

Expected output should include:

```text
GStreamer version: GStreamer 1.28.2
webrtcbin: ok
appsink: ok
appsrc: ok
decodebin: ok
videoconvert: ok
audioconvert: ok
audioresample: ok
opusdec: ok
vp8dec: ok
rtpbin: ok
```

The probe validates:

- MSVC import library linking
- runtime DLL discovery
- GStreamer plugin discovery
- availability of WebRTC, appsrc/appsink, decode, video, audio, Opus, VP8, and RTP elements

Keep GStreamer/WebRTC media code under `src/media`; `src/net` should remain the
HTTP/WebSocket control/result runtime.
