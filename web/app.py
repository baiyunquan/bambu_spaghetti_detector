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


def create_app(snapshot_dir, sampler=None):
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
