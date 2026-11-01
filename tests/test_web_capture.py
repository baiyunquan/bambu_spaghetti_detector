from fastapi.testclient import TestClient
import pytest

from test_web_gallery import Sampler


class Capture:
    def __init__(self, error=None): self.calls = 0; self.error = error
    def request(self, timeout=10):
        self.calls += 1
        if self.error: raise self.error
        return 'manual_test.jpg'


def test_capture_button_and_post_returns_local_detail_url(tmp_path):
    from web.app import create_app
    capture = Capture()
    with TestClient(create_app(tmp_path, sampler=Sampler(), capture=capture)) as browser:
        page = browser.get('/photos').text
        assert '拍一张' in page
        assert '/static/capture.js' in page
        result = browser.post('/api/photos/capture', headers={'X-Capture-Request': '1'})
        assert result.status_code == 201
        assert result.json() == {'filename': 'manual_test.jpg', 'url': '/photos/manual_test.jpg'}
        assert capture.calls == 1


def test_cross_origin_or_plain_form_cannot_take_photo(tmp_path):
    from web.app import create_app
    capture = Capture()
    with TestClient(create_app(tmp_path, sampler=Sampler(), capture=capture)) as browser:
        assert browser.post('/api/photos/capture').status_code == 403
        assert browser.post('/api/photos/capture', headers={
            'X-Capture-Request': '1', 'Origin': 'https://foreign.example'}).status_code == 403
        assert capture.calls == 0


@pytest.mark.parametrize('name,status', [('CaptureBusy',409), ('CaptureUnavailable',503), ('TimeoutError',504), ('OSError',507)])
def test_capture_failure_is_reported(tmp_path, name, status):
    from web.app import create_app
    from api.frame_recording import CaptureBusy, CaptureUnavailable
    errors = {'CaptureBusy': CaptureBusy(), 'CaptureUnavailable': CaptureUnavailable(),
              'TimeoutError': TimeoutError(), 'OSError': OSError()}
    with TestClient(create_app(tmp_path, sampler=Sampler(), capture=Capture(errors[name]))) as browser:
        response = browser.post('/api/photos/capture', headers={'X-Capture-Request':'1'})
        assert response.status_code == status


def test_standalone_gallery_reports_capture_unavailable(tmp_path):
    from web.app import create_app
    with TestClient(create_app(tmp_path, sampler=Sampler())) as browser:
        assert browser.post('/api/photos/capture', headers={'X-Capture-Request':'1'}).status_code == 503
