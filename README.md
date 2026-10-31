# Bambu Spaghetti Failure Detection & Print Halting System

Real-time spaghetti failure detection and print halting system for Bambu Lab A1 3D printer running on NVIDIA Jetson Nano.

## Architecture
- `api/`: Main controller and Bambu API service (Python 3.10)
- `inference/`: GPU inference worker with TensorRT FP16 (JetPack Python 3.6)
- `scripts/`: Utility scripts (Access Code retrieval, model conversion)
