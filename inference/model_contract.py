"""Versioned, hash-bound engine metadata; imports no inference/GPU runtime."""
import hashlib
import json
import numbers
import os
import re
import tempfile

SCHEMA_VERSION = 1
CANONICAL_LABEL_MAP_VERSION = 1
DEFAULT_CLASS_NAMES = {
    'obico': ['spaghetti'],
    'yolov8': ['Spaghetti - v11 2025-04-15 12-28am', 'warping'],
    'yolov5': ['error', 'extrusor', 'part', 'spaghetti'],
}
CANONICAL_LABEL_MAP = {'Spaghetti - v11 2025-04-15 12-28am': 'spaghetti'}
_FIELDS = {'schema_version', 'engine_sha256', 'model_type', 'class_names'}


def validate_model_type(model_type):
    if model_type not in DEFAULT_CLASS_NAMES:
        raise ValueError('Unsupported model_type: {}'.format(model_type))
    return model_type


def validate_class_names(class_names):
    if (not isinstance(class_names, list) or not class_names or
            any(not isinstance(name, str) or not name.strip() for name in class_names) or
            len(set(class_names)) != len(class_names)):
        raise ValueError('class_names must be a nonempty list of unique nonempty strings')
    return list(class_names)


def canonicalize_labels(class_names):
    """Map only known raw export labels, retaining class IDs and other names."""
    return [CANONICAL_LABEL_MAP.get(name, name) for name in validate_class_names(class_names)]


def engine_sha256(engine_path):
    digest = hashlib.sha256()
    with open(engine_path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_metadata_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate metadata field: {}'.format(key))
        result[key] = value
    return result


def write_metadata(engine_path, model_type, class_names=None):
    """Atomically write metadata alongside an engine (including symlink paths)."""
    validate_model_type(model_type)
    names = validate_class_names(DEFAULT_CLASS_NAMES[model_type] if class_names is None else class_names)
    if model_type == 'obico' and len(names) != 1:
        raise ValueError('Obico requires exactly one class')
    metadata = dict(schema_version=SCHEMA_VERSION, engine_sha256=engine_sha256(engine_path),
                    model_type=model_type, class_names=names)
    path = os.fspath(engine_path) + '.meta.json'
    descriptor, temporary = tempfile.mkstemp(prefix='.model-meta-', dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(descriptor, 'w') as target:
            json.dump(metadata, target, indent=2, sort_keys=True)
            target.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def _static_shape(shape):
    try:
        result = tuple(shape)
    except TypeError:
        raise ValueError('Bindings must have static positive integer shapes')
    if any(isinstance(dim, bool) or not isinstance(dim, numbers.Integral) or dim <= 0 for dim in result):
        raise ValueError('Bindings must have static positive integer shapes')
    return result


def validate_shapes(input_shape, output_shapes, model_type, class_count):
    """Validate explicit NCHW and raw output layouts; never infer a YOLO type."""
    validate_model_type(model_type)
    shape = _static_shape(input_shape)
    if len(shape) != 4 or shape[:2] != (1, 3):
        raise ValueError('Input binding must have shape [1,3,H,W]')
    outputs = [_static_shape(value) for value in output_shapes]
    if model_type == 'obico':
        valid = (class_count == 1 and len(outputs) == 2 and len(outputs[0]) == 4 and
                 outputs[0][0] == 1 and outputs[0][2:] == (1, 4) and
                 outputs[1] == (1, outputs[0][1], 1))
    elif model_type == 'yolov8':
        valid = len(outputs) == 1 and len(outputs[0]) == 3 and outputs[0][:2] == (1, 4 + class_count)
    else:
        valid = (len(outputs) == 1 and len(outputs[0]) == 3 and
                 outputs[0][0] == 1 and outputs[0][2] == 5 + class_count)
    if not valid:
        raise ValueError('Invalid {} output bindings or metadata class count: {}'.format(model_type, outputs))


class ModelContract:
    """Fail closed if a sidecar does not describe exactly this engine and type."""
    def __init__(self, engine_path, model_type):
        self.engine_path = os.fspath(engine_path)
        self.model_type = validate_model_type(model_type)
        self.metadata_path = self.engine_path + '.meta.json'
        try:
            with open(self.metadata_path) as source:
                metadata = json.load(source, object_pairs_hook=_unique_metadata_fields)
        except (OSError, ValueError) as exc:
            raise ValueError('Cannot read engine metadata: {}'.format(exc))
        if not isinstance(metadata, dict) or set(metadata) != _FIELDS:
            raise ValueError('Engine metadata fields do not match schema version 1')
        if type(metadata['schema_version']) is not int or metadata['schema_version'] != SCHEMA_VERSION:
            raise ValueError('Unsupported engine metadata schema_version')
        if metadata['model_type'] != model_type:
            raise ValueError('Engine metadata model_type mismatch')
        names = validate_class_names(metadata['class_names'])
        if model_type == 'obico' and len(names) != 1:
            raise ValueError('Obico requires exactly one class')
        checksum = metadata['engine_sha256']
        if not isinstance(checksum, str) or not re.fullmatch('[0-9a-f]{64}', checksum):
            raise ValueError('Invalid engine SHA256 metadata')
        try:
            actual = engine_sha256(self.engine_path)
        except OSError as exc:
            raise ValueError('Cannot hash engine: {}'.format(exc))
        if checksum != actual:
            raise ValueError('Engine SHA256 does not match metadata')
        self.metadata = metadata
        self.class_names = names
        self.canonical_labels = canonicalize_labels(names)

    def validate_bindings(self, input_shape, output_shapes):
        validate_shapes(input_shape, output_shapes, self.model_type, len(self.class_names))
