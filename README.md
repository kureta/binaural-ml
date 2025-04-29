# Real-Time Binaural Trajectory Estimation & Visualization

This project provides a high-speed, low-latency system for:

- Streaming stereo audio (microphone)
- Neural model inference (GPU) for 3D trajectory estimation
- Real-time 3D visualization with trajectory fading

---

## Components

- **server_inference/**: C++ application using Libtorch for real-time GPU inference
- **client_visualizer/**: Python app to plot real-time 3D position trail
- **scripts/**: TorchScript export tools if you want to retrain/update models
- **utils/**: Optional data generators for synthetic training

---

## Setup Instructions

1. Install libtorch C++:
   [pytorch get started locally](https://pytorch.org/get-started/locally/)

2. Install Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Build C++ Inference Server

   ```bash
   cd server_inference
   mkdir build && cd build
   cmake ..
   make
   ./streaming_infer_server
   ```

4. Run Python Visualizer

   ```bash
   cd client_visualizer
   python live_3d_visualizer.py
   ```

---

## Notes

- Model assumes stereo 48kHz microphone input.
- Server auto sends estimated XYZ via TCP.
- Client plots smoothed, faded 3D trajectory in real-time.

---

## License

MIT License
