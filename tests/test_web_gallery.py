import json
import os
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]


class Sampler:
    def start(self): pass
    def stop(self): pass
    def snapshot(self):
        return {'status': 'unavailable', 'current': None, 'samples': [], 'message': '采集不可用'}


def client(root):
    assert (ROOT / 'web/app.py').exists(), 'Web application not implemented'
    from web.app import create_app
    return TestClient(create_app(root, sampler=Sampler()))


def photo(root, name='failure_20261101_100000.jpg', label='spaghetti'):
    path=root/name
    Image.new('RGB',(64,40),(210,210,210)).save(path)
    path.with_suffix('.json').write_text(json.dumps({
        'timestamp':'20261101_100000','consecutive_hits':6,
        'detections':[{'label':label,'class_id':3,'confidence':.88,'bbox':[1,2,30,35]}]
    }))
    return path


def test_empty_and_missing_directories(tmp_path):
    for root in [tmp_path,tmp_path/'not-created']:
        with client(root) as browser:
            assert browser.get('/').headers['content-type'].startswith('text/html')
            assert '暂无照片' in browser.get('/photos').text
            assert browser.get('/jetson').status_code==200
            assert browser.get('/api/jetson').json()['status']=='unavailable'
        assert not (tmp_path/'not-created').exists()


def test_24_per_page_newest_first_and_detail(tmp_path):
    for i in range(26):
        path=photo(tmp_path,'image_{:02}.jpg'.format(i))
        os.utime(path,(1000+i,1000+i))
    with client(tmp_path) as browser:
        first=browser.get('/photos').text
        assert first.count('data-photo=')==24
        assert first.index('image_25.jpg')<first.index('image_24.jpg')
        assert 'image_00.jpg' not in first
        assert browser.get('/photos?page=2').text.count('data-photo=')==2
        assert browser.get('/photos?page=0').status_code==422
        assert browser.get('/photos?page=3').status_code==404
        detail=browser.get('/photos/image_25.jpg')
        assert detail.status_code==200
        for text in ['spaghetti','88.0%','连续命中','1, 2, 30, 35']:
            assert text in detail.text
        image=browser.get('/media/image_25.jpg')
        assert image.content==(tmp_path/'image_25.jpg').read_bytes()
        assert image.headers['content-type']=='image/jpeg'


@pytest.mark.parametrize('broken',[None,'{','[]','{"detections":[{"confidence":"nan"}]}'])
def test_missing_or_bad_metadata_does_not_hide_image(tmp_path,broken):
    path=photo(tmp_path)
    if broken is None:path.with_suffix('.json').unlink()
    else:path.with_suffix('.json').write_text(broken)
    with client(tmp_path) as browser:
        assert '检测记录不可用' in browser.get('/photos/'+path.name).text
        assert browser.get('/media/'+path.name).status_code==200


def test_label_is_escaped_and_all_assets_are_local(tmp_path):
    path=photo(tmp_path,label='<script>alert("bad")</script>')
    with client(tmp_path) as browser:
        text=browser.get('/photos/'+path.name).text
        assert '<script>alert(' not in text
        assert '&lt;script&gt;' in text
        assert 'https://' not in text and 'http://' not in text
        css=browser.get('/static/app.css')
        assert css.status_code==200 and '--bs-border-radius: 0' in css.text
        assert browser.get('/static/vendor/bootstrap.min.css').status_code==200
        assert browser.get('/static/vendor/chart.umd.min.js').status_code==200


def test_paths_and_external_symlinks_are_not_served(tmp_path):
    root=tmp_path/'photos';root.mkdir()
    secret=photo(tmp_path,'secret.jpg')
    (root/'escape.jpg').symlink_to(secret)
    normal=photo(root)
    normal.with_suffix('.json').unlink()
    normal.with_suffix('.json').symlink_to(secret.with_suffix('.json'))
    with client(root) as browser:
        for url in ['/media/escape.jpg','/photos/escape.jpg','/media/%2e%2e%2fsecret.jpg',
                    '/media/../secret.jpg','/media/config.json']:
            assert browser.get(url).status_code==404,url
        assert 'escape.jpg' not in browser.get('/photos').text
        assert '检测记录不可用' in browser.get('/photos/'+normal.name).text
        assert browser.post('/api/jetson').status_code==405
