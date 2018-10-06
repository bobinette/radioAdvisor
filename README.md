# radioAdvisor

## On Mac

Using python 3
```bash
python3 -m venv ./env
./env/bin/activate
pip install -U pip
pip install ipython numpy nibabel matplotlib
```

You'll need to install opencv, this tutorial explains it pretty well: https://www.learnopencv.com/install-opencv3-on-macos/

Notes:
- the `find` command did not work but the `.so` file was at `/usr/local/opt/opencv@3/lib/python3.7/site-packages/cv2.cpython-37m-darwin.so`
- the lib I got installed is for python 3.7 but my env runs python 3.6, still everything is working well.

## Annotating the images

To annotate the images, use
```bash
python draw_box.py <first letter of your first name>
```
