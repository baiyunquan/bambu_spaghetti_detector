"""Jinja pages and bounded telemetry, independent of GPU/printer APIs."""
from contextlib import asynccontextmanager
from pathlib import Path
from anyio.to_thread import current_default_thread_limiter

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .gallery import PhotoStore

HERE = Path(__file__).resolve().parent


def create_app(snapshot_dir, sampler=None, capture=None):
    if sampler is None:
        from .telemetry import TelemetrySampler
        sampler = TelemetrySampler()
    store = PhotoStore(snapshot_dir)
    templates = Jinja2Templates(directory=str(HERE / 'templates'))

    @asynccontextmanager
    async def lifespan(app):
        limiter = current_default_thread_limiter()
        previous_limit = limiter.total_tokens
        limiter.total_tokens = 4
        try:
            sampler.start()
            yield
        finally:
            try:
                sampler.stop()
            finally:
                limiter.total_tokens = previous_limit

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.mount('/static', StaticFiles(directory=str(HERE / 'static')), name='static')

    @app.get('/')
    def home():
        return RedirectResponse('/photos')

    @app.get('/photos')
    def photos(request: Request, page: int = Query(1, ge=1)):
        try:
            data = store.list_page(page)
        except FileNotFoundError:
            raise HTTPException(404, '照片页不存在')
        except OSError:
            return templates.TemplateResponse(request=request, name='photos.html',
                    context={'tab':'photos','photos':[],'error':'照片目录暂不可用','page':1,'pages':1,'total':0}, status_code=503)
        return templates.TemplateResponse(request=request, name='photos.html', context={'tab':'photos',**data})

    @app.get('/photos/{filename}')
    def detail(request: Request, filename: str):
        try:
            data = store.detail(filename)
        except OSError:
            raise HTTPException(404, '照片不存在')
        return templates.TemplateResponse(request=request, name='detail.html', context={'tab':'photos',**data})

    @app.post('/api/photos/capture', status_code=201)
    def take_photo(request: Request):
        from api.frame_recording import CaptureBusy, CaptureUnavailable
        # A custom header prevents cross-site forms; Origin also protects browser requests.
        origin = request.headers.get('origin')
        expected_origin = '{}://{}'.format(request.url.scheme, request.headers.get('host', ''))
        if request.headers.get('x-capture-request') != '1' or (origin and origin != expected_origin):
            raise HTTPException(403, '请从本站照片页发起拍摄')
        if capture is None:
            raise HTTPException(503, '相机采集尚未就绪')
        try:
            filename = capture.request(timeout=10)
        except CaptureBusy as exc:
            raise HTTPException(409, str(exc) or '已有拍摄请求')
        except CaptureUnavailable as exc:
            raise HTTPException(503, str(exc) or '相机采集不可用')
        except TimeoutError as exc:
            raise HTTPException(504, str(exc) or '拍摄超时，请检查相机连接')
        except OSError as exc:
            raise HTTPException(507, '照片保存失败，请检查磁盘空间和目录权限')
        return {'filename': filename, 'url': request.url_for('detail', filename=filename).path}

    @app.get('/media/{filename}')
    def image(filename: str):
        try:
            path = store.open_image(filename)
        except OSError:
            raise HTTPException(404, '照片不存在')
        return FileResponse(path, headers={'X-Content-Type-Options':'nosniff'})

    @app.get('/jetson')
    def jetson(request: Request):
        return templates.TemplateResponse(request=request, name='jetson.html', context={'tab':'jetson'})

    @app.get('/api/jetson')
    def metrics():
        return JSONResponse(sampler.snapshot(), headers={'Cache-Control':'no-store'})

    return app
