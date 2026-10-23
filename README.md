# Mall People Tracker — Footfall & People Counting

Real-time people tracking and counting for a shopping-mall entrance or section, built with **YOLOv8** and **OpenCV**. The system watches a virtual line (a doorway), tracks every person with a persistent ID, and counts crossings in both directions — **IN** and **OUT** — with a live occupancy estimate (**IN − OUT**). This is the core of a footfall-analytics system a mall would use to measure visitor traffic per entrance, store zone, or time of day.

## Features
- YOLOv8 person detection with persistent track IDs (ByteTrack via `model.track`, or the built-in centroid tracker)
- Direction-aware counting: crossing the line left → right = **IN**, right → left = **OUT**, with a deadband so someone standing on the line is never double-counted
- Live overlay: bounding boxes with IDs, IN / OUT counters, current occupancy, FPS, and a flash indicator each time someone is counted
- **Simulation mode** — a synthetic mall-entrance scene rendered with OpenCV/NumPy that runs the *same* tracking, counting and overlay pipeline, so the project can be demoed anywhere with no camera and no model weights
- Works on video files or webcam; the annotated video can be saved with `--output`

## How the Counting Works
1. **Detect** — YOLOv8 finds people in each frame (the simulation substitutes ground-truth boxes from its synthetic scene; detections are an interchangeable source by design).
2. **Track** — each person gets a stable ID (ByteTrack in real mode; a centroid tracker otherwise) so the same person is followed across frames.
3. **Cross** — a virtual vertical line marks the entrance. When a track's centroid moves from one side of the line to the other, it is counted once, in that direction.
4. **Count** — occupancy at any moment is `IN − OUT`, shown live on the HUD together with the totals.

## Tech Stack
Python • YOLOv8 (Ultralytics) • OpenCV • NumPy • ByteTrack

## How to Run
```bash
pip install -r requirements.txt
```

**Simulation mode** (only needs `opencv-python` + `numpy` — no GPU, no model download):
```bash
python people_counter.py --simulate                                   # preview window, 30 s
python people_counter.py --simulate --output demo.mp4 --duration 30   # save the demo video
python people_counter.py --simulate --line-x 0.4 --max-people 9       # move the line, busier mall
```

**Real mode** (needs `ultralytics`, installed by the requirements file):
```bash
python people_counter.py --source mall_entrance.mp4                   # video file
python people_counter.py --source 0                                   # webcam
python people_counter.py --source mall_entrance.mp4 --output counted.mp4
python people_counter.py --source mall_entrance.mp4 --tracker centroid  # IDs from the built-in tracker instead of ByteTrack
```

Useful options: `--conf 0.4` (detection threshold), `--model yolov8n.pt` (any YOLOv8 weights), `--line-x 0.5` (line position as a fraction of the frame width), `--no-display` (headless). Press `q` or `Esc` to stop.

## Demo
The demo video for this project (`mall-people-tracker-demo.mp4`) is the **simulation mode rendered through the actual pipeline** — the synthetic scene produces the frames and ground-truth boxes, and the real tracker, line-crossing counter and overlay do the rest. It is a simulation, not real mall footage; the same code counts real footage unchanged in real mode.

## Real-World Uses
- Footfall counting at mall / retail-store entrances (visitors per hour, per entrance)
- Occupancy monitoring for safety limits and crowd management
- Conversion analytics when combined with sales data (visitors vs. buyers)
- Queue and staffing decisions based on peak-hour traffic

## Project Structure
```
mall-people-tracker/
├── people_counter.py   # main script (real mode + simulation mode)
├── requirements.txt
└── README.md
```

## Future Improvements
- Overhead (top-down) camera support with a horizontal counting line and multi-line / multi-zone counting
- Hourly traffic reports exported to CSV, plus a simple daily footfall chart
- Re-identification–free dwell-time estimation per zone
- Optional DeepSORT/BoT-SORT backends for crowded scenes

## Author
Abdul Hanan — Computer Vision & AI Developer
GitHub: https://github.com/alishan24456-glitch
LinkedIn: https://www.linkedin.com/in/ali-raza-computervision/
