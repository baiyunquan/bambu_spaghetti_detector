#!/usr/bin/env python3
"""Write hash-bound metadata after TensorRT compilation, without GPU imports."""
import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.model_contract import write_metadata


def onnx_class_names(path):
    """Load the exporter names entry as data, never as executable Python."""
    import onnx
    model = onnx.load(path, load_external_data=False)
    entries = [item.value for item in model.metadata_props if item.key == 'names']
    if len(entries) != 1:
        raise ValueError('ONNX must contain exactly one names metadata entry')
    names = ast.literal_eval(entries[0])
    if isinstance(names, dict):
        if set(names) != set(range(len(names))) or any(type(key) is not int for key in names):
            raise ValueError('ONNX class IDs must be contiguous integers starting at zero')
        names = [names[index] for index in range(len(names))]
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--type', required=True, choices=['obico', 'yolov8', 'yolov5'])
    parser.add_argument('--onnx', help='Host ONNX with original names metadata; otherwise use known model labels')
    args = parser.parse_args()
    try:
        names = onnx_class_names(args.onnx) if args.onnx else None
        path = write_metadata(args.engine, args.type, names)
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))
    print(path)


if __name__ == '__main__':
    main()
