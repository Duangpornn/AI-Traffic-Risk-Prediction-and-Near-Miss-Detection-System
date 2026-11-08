# AI Traffic Risk Prediction

Standalone Streamlit app for uploading traffic video, detecting objects with a
trained Ultralytics YOLO model, tracking them with ByteTrack, screening for
image-space near-misses, and downloading an annotated result video. The
interface supports Thai and English; use the language selector at the top of
the page to switch languages. Analyzed videos include a whole-video centroid
density heatmap and the last 30 points of each persistent vehicle track, using
the same Gaussian blur, JET color map, and overlay weights as the final video
cell in the training notebook.

## Run locally

From this directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Put the trained weights at `models\thai_vehicle_yolov8_best.pt`, or set
`TRAFFIC_MODEL_PATH` to the `.pt` file's location before starting Streamlit.
The supplied model is stored at `models\thai_vehicle_yolov8_best.pt`. Keep the
weights available at that path when running the app.

The analyzed video is encoded as browser-compatible H.264 MP4. This uses the
bundled FFmpeg executable provided by `imageio-ffmpeg`.

## Risk estimate limitations

The app estimates time to closest approach using tracked bounding-box centers
and velocity in image pixels. Without camera calibration, perspective handling,
and world-coordinate conversion, this is a screening heuristic rather than
physically calibrated TTC. It does not replace a validated road-safety system.

The application currently processes each upload in the Streamlit server's
local temporary storage. For cloud deployment, provide the model file through
an appropriate private artifact or mounted storage; do not commit model weights
or credentials to source control.
