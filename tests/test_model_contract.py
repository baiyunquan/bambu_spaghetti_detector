"""CPU-only oracle tests for engine identity, labels, and native decode gates."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.model_contract import ModelContract, write_metadata
from inference.reference import decode, decode_candidates, reference_nms

META = dict(orig_w=200, orig_h=100, target_w=100, target_h=100,
            scale=0.5, pad_x=0, pad_y=25)


@pytest.fixture
def engine(tmp_path):
    path = tmp_path / 'model.engine'
    path.write_bytes(b'engine-identity')
    return path


def test_metadata_sha_labels_and_symlink(engine, tmp_path):
    link = tmp_path / 'linked.engine'
    link.symlink_to(engine)
    path = write_metadata(str(link), 'yolov8')
    assert Path(path) == Path(str(link) + '.meta.json')
    contract = ModelContract(str(link), 'yolov8')
    assert contract.class_names == ['Spaghetti - v11 2025-04-15 12-28am', 'warping']
    assert contract.canonical_labels == ['spaghetti', 'warping']
    assert json.loads(Path(path).read_text())['engine_sha256'] == hashlib.sha256(b'engine-identity').hexdigest()
    engine.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='SHA256'):
        ModelContract(str(link), 'yolov8')


@pytest.mark.parametrize('patch', [
    {'schema_version': 2}, {'schema_version': True}, {'engine_sha256': '0'},
    {'model_type': 'obico'}, {'class_names': []}, {'class_names': ['x', 'x']},
    {'class_names': ['']}, {'class_names': [1]}, {'unknown': 'value'},
])
def test_metadata_fails_closed(engine, patch):
    path = write_metadata(str(engine), 'yolov8')
    content = json.loads(Path(path).read_text())
    content.update(patch)
    Path(path).write_text(json.dumps(content))
    with pytest.raises(ValueError):
        ModelContract(str(engine), 'yolov8')


def test_metadata_missing_rejected(engine):
    with pytest.raises(ValueError, match='metadata'):
        ModelContract(str(engine), 'obico')


@pytest.mark.parametrize('model,outputs', [
    ('obico', [(1, 4, 1, 4), (1, 4, 1)]),
    ('yolov8', [(1, 6, 4)]), ('yolov5', [(1, 4, 9)]),
])
def test_shapes_accept_exact_layouts(engine, model, outputs):
    write_metadata(str(engine), model)
    c = ModelContract(str(engine), model)
    c.validate_bindings((1, 3, 416, 640), outputs)
    for shape in [(2, 3, 416, 640), (1, 1, 416, 640), (1, 3, -1, 640), (1, 3, 0, 640)]:
        with pytest.raises(ValueError):
            c.validate_bindings(shape, outputs)
    with pytest.raises(ValueError):
        c.validate_bindings((1, 3, 416, 640), [(1, 2, 3)])


def test_binding_class_count_and_obico_anchor_count(engine):
    write_metadata(str(engine), 'yolov8', ['one', 'two', 'three'])
    c = ModelContract(str(engine), 'yolov8')
    with pytest.raises(ValueError):
        c.validate_bindings((1, 3, 640, 640), [(1, 6, 10)])
    write_metadata(str(engine), 'obico')
    c = ModelContract(str(engine), 'obico')
    with pytest.raises(ValueError):
        c.validate_bindings((1, 3, 416, 416), [(1, 4, 1, 4), (1, 3, 1)])


def test_cli_writes_metadata_from_defaults(engine):
    completed = subprocess.run([sys.executable, str(ROOT / 'scripts/write_model_metadata.py'),
                                '--engine', str(engine), '--type', 'yolov5'], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    c = ModelContract(str(engine), 'yolov5')
    assert c.canonical_labels == ['error', 'extrusor', 'part', 'spaghetti']


def test_obico_scale_padding_and_threshold_boundary():
    outputs = [np.array([[[[0.1, 0.25, 0.9, 0.75]], [[0, 0, 1, 1]]]]),
               np.array([[[0.5], [0.499]]])]
    assert decode(outputs, 'obico', META, 0.5, 0.5, ['spaghetti']) == [
        dict(label='spaghetti', class_id=0, confidence=0.5, bbox=[20, 0, 180, 100])]
    candidates = decode_candidates(outputs, 'obico', META, 0.5)
    np.testing.assert_allclose(candidates, [[10, 25, 90, 75, .5, 0, 0]])


def test_v8_single_class_per_anchor_classwise_nms_and_raw_label_mapping():
    # First two anchors overlap but belong to distinct classes; anchor 2
    # duplicates anchor 0 at equal score and must lose by original index.
    preds = np.array([[50, 50, 40, 20, .8, .2],
                      [50, 50, 40, 20, .1, .9],
                      [50, 50, 40, 20, .8, .8]], dtype=np.float32)
    result = decode([preds.T[None]], 'yolov8', META, .5, .45,
                    ['Spaghetti - v11 2025-04-15 12-28am', 'warping'])
    assert result == [dict(label='warping', class_id=1, confidence=.9, bbox=[60, 30, 140, 70]),
                      dict(label='spaghetti', class_id=0, confidence=.8, bbox=[60, 30, 140, 70])]


def test_v5_multiplies_objectness_and_preserves_nonfailure_class_ids():
    preds = np.array([[[50, 50, 40, 20, .5, .9, .1, .2, .3],
                       [20, 40, 10, 10, 1, .1, .2, .8, .3]]])
    result = decode([preds], 'yolov5', META, .45, .45,
                    ['error', 'extrusor', 'part', 'spaghetti'])
    assert [(d['label'], d['class_id'], d['confidence']) for d in result] == [('part', 2, .8), ('error', 0, .45)]
    np.testing.assert_allclose(decode_candidates([preds], 'yolov5', META, .45)[:, 4:], [[.45, 0, 0], [.8, 2, 1]])


def test_nms_boundary_ties_classes_and_no_candidate_cap():
    # Intersection 2, union 4 => IoU exactly 0.5: equality is retained.
    candidates = np.array([[0, 0, 3, 1, .8, 0, 7], [1, 0, 4, 1, .8, 0, 2],
                           [1, 0, 4, 1, .7, 1, 8]], dtype=float)
    assert reference_nms(candidates, .5).tolist() == [1, 0, 2]
    assert reference_nms(candidates, .49).tolist() == [1, 2]
    many = np.zeros((1200, 7))
    many[:, 0] = np.arange(1200) * 2
    many[:, 2] = many[:, 0] + 1
    many[:, 3:5] = 1
    many[:, 6] = np.arange(1200)
    assert len(reference_nms(many, .45)) == 1200


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_predictions_rejected_even_below_threshold(bad):
    preds = np.array([[[50., 50, 40, 20, .1, .2]]]).transpose(0, 2, 1)
    preds[0, 0, 0] = bad
    with pytest.raises(ValueError, match='finite'):
        decode([preds], 'yolov8', META, .5, .45, ['spaghetti', 'warping'])


def test_bad_shapes_thresholds_metadata_and_labels_rejected():
    outputs = [np.zeros((1, 6, 1))]
    for kwargs in [dict(conf_thresh=float('nan')), dict(iou_thresh=1.1),
                   dict(meta=dict(META, scale=0)), dict(meta=dict(META, pad_x=float('inf'))),
                   dict(class_names=['onlyone'])]:
        args = dict(raw_outputs=outputs, model_type='yolov8', meta=META,
                    conf_thresh=.5, iou_thresh=.45, class_names=['spaghetti', 'warping'])
        args.update(kwargs)
        with pytest.raises(ValueError):
            decode(**args)
    with pytest.raises(ValueError):
        decode([np.zeros((1, 10, 6))], 'yolov8', META, .5, .45, ['a', 'b'])


def test_empty_outputs_are_empty_and_box_clipping_uses_original_frame():
    assert decode([np.zeros((1, 6, 0))], 'yolov8', META, .5, .45, ['a', 'b']) == []
    pred = np.array([[[50], [50], [400], [400], [.55555], [.1]]])
    assert decode([pred], 'yolov8', META, .5, .45, ['a', 'b']) == [
        dict(label='a', class_id=0, confidence=.5555, bbox=[0, 0, 200, 100])]


def test_duplicate_metadata_fields_rejected(engine):
    path = write_metadata(str(engine), 'obico')
    content = Path(path).read_text()
    Path(path).write_text(content.replace('{', '{"schema_version": 1,', 1))
    with pytest.raises(ValueError, match='metadata'):
        ModelContract(str(engine), 'obico')


@pytest.mark.parametrize('names,success', [
    ("{1: 'custom_failure', 0: 'Spaghetti - v11 2025-04-15 12-28am'}", True),
    ("{1: 'bad_gap'}", False),
    ("__import__('os').getcwd()", False),
    ("{'0': 'bad_string_id'}", False),
])
def test_cli_reads_safe_original_onnx_labels(engine, tmp_path, names, success):
    onnx = pytest.importorskip('onnx')
    model = onnx.helper.make_model(onnx.helper.make_graph([], 'metadata-only', [], []))
    entry = model.metadata_props.add()
    entry.key, entry.value = 'names', names
    path = tmp_path / 'source.onnx'
    onnx.save(model, str(path))
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/write_model_metadata.py'),
                             '--engine', str(engine), '--type', 'yolov8', '--onnx', str(path)],
                            capture_output=True, text=True)
    if success:
        assert result.returncode == 0, result.stderr
        contract = ModelContract(str(engine), 'yolov8')
        assert contract.class_names == ['Spaghetti - v11 2025-04-15 12-28am', 'custom_failure']
        assert contract.canonical_labels == ['spaghetti', 'custom_failure']
    else:
        assert result.returncode != 0
        assert not Path(str(engine) + '.meta.json').exists()
