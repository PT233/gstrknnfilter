# GStreamer RKNN Plugin

GStreamer 插件，支持在 RKNN NPU 上运行多种视觉模型（YOLO、分类、RetinaFace 等）。

## 构建

推荐使用脚本（避免 meson 弃用警告）：

    ./build.sh

或手动执行：

    meson setup build && ninja -C build

## 测试

在 `gstreamer-rknn` 目录下运行（使用 model/ 下的 yolov5s 和 coco 标签）：

    ./run_test.sh

或手动指定**绝对路径**（不要用 /model/，那是根目录）：

    gst-launch-1.0 videotestsrc num-buffers=300 ! video/x-raw,format=NV12,width=640,height=480 ! \
      rknnfilter model-path=/userdata/root_backup/gstreamer-rknn/model/yolov5s-640-640.rknn \
        model-type=yolov5 label-path=/userdata/root_backup/gstreamer-rknn/model/coco_80_labels_list.txt \
        show-fps=true ! videoconvert ! autovideosink

* gst-app :
  basic meson-based layout for writing a GStreamer-based application.

* gst-plugin :
  basic meson-based layout and basic filter code for writing a GStreamer plug-in.

## License

This code is provided under a MIT license [MIT], which basically means "do
with it as you wish, but don't blame us if it doesn't work". You can use
this code for any project as you wish, under any license as you wish. We
recommend the use of the LGPL [LGPL] license for applications and plugins,
given the minefield of patents the multimedia is nowadays. See our website
for details [Licensing].

## Usage

Configure and build all examples (application and plugins) as such:

    meson builddir
    ninja -C builddir

See <https://mesonbuild.com/Quick-guide.html> on how to install the Meson
build system and ninja.

Modify `gst-plugin/meson.build` to add or remove source files to build or
add additional dependencies or compiler flags or change the name of the
plugin file to be installed.

Modify `meson.build` to check for additional library dependencies
or other features needed by your plugin.

Once the plugin is built you can either install system-wide it with `sudo ninja
-C builddir install` (however, this will by default go into the `/usr/local`
prefix where it won't be picked up by a `GStreamer` installed from packages, so
you would need to set the `GST_PLUGIN_PATH` environment variable to include or
point to `/usr/local/lib/gstreamer-1.0/` for your plugin to be found by a
from-package `GStreamer`).

Alternatively, you will find your plugin binary in `builddir/gst-plugins/src/`
as `libgstplugin.so` or similar (the extension may vary), so you can also set
the `GST_PLUGIN_PATH` environment variable to the `builddir/gst-plugins/src/`
directory (best to specify an absolute path though).

You can also check if it has been built correctly with:

    gst-inspect-1.0 builddir/gst-plugins/src/libgstplugin.so

## Auto-generating your own plugin

You will find a helper script in `gst-plugins/tools/make_element` to generate
the source/header files for a new plugin.

To create sources for `myfilter` based on the `gsttransform` template run:

``` shell
cd src;
../tools/make_element myfilter gsttransform
```

This will create `gstmyfilter.c` and `gstmyfilter.h`. Open them in an editor and
start editing. There are several occurances of the string `template`, update
those with real values. The plugin will be called `myfilter` and it will have
one element called `myfilter` too. Also look for `FIXME:` markers that point you
to places where you need to edit the code.

You can then add your sources files to `gst-plugins/meson.build` and re-run
ninja to have your plugin built.


[MIT]: http://www.opensource.org/licenses/mit-license.php or COPYING.MIT
[LGPL]: http://www.opensource.org/licenses/lgpl-license.php or COPYING.LIB
[Licensing]: https://gstreamer.freedesktop.org/documentation/application-development/appendix/licensing.html
