#include <gst/gst.h>

#include <array>
#include <iostream>

int main(int argc, char** argv) {
    gst_init(&argc, &argv);

    std::cout << "GStreamer version: "
              << gst_version_string()
              << "\n";

    constexpr std::array<const char*, 10> elements = {
        "webrtcbin",
        "appsink",
        "appsrc",
        "decodebin",
        "videoconvert",
        "audioconvert",
        "audioresample",
        "opusdec",
        "vp8dec",
        "rtpbin",
    };

    bool ok = true;
    for (const auto* name : elements) {
        GstElementFactory* factory = gst_element_factory_find(name);
        std::cout << name << ": " << (factory ? "ok" : "missing") << "\n";
        if (factory) {
            gst_object_unref(factory);
        } else {
            ok = false;
        }
    }

    return ok ? 0 : 1;
}
