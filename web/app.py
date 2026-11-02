"""Jinja pages and bounded telemetry, independent of GPU/printer APIs."""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from anyio.to_thread import current_default_thread_limiter

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .gallery import PhotoStore

HERE = Path(__file__).resolve().parent


def create_app(snapshot_dir, sampler=None, capture=None, mqtt_controller=None,
               config_path="config.json", on_reconnect=None):
    if sampler is None:
        from .telemetry import TelemetrySampler
        sampler = TelemetrySampler()
    store = PhotoStore(snapshot_dir)
    templates = Jinja2Templates(directory=str(HERE / 'templates'))

    get_mqtt = mqtt_controller if callable(mqtt_controller) else (lambda: mqtt_controller)

    def is_printer_connected() -> bool:
        ctrl = get_mqtt()
        return bool(ctrl is not None and getattr(ctrl, "is_connected", False))

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
        if is_printer_connected():
            return RedirectResponse('/status')
        return RedirectResponse('/login?reason=disconnected')

    @app.get('/status')
    def status_page(request: Request):
        return templates.TemplateResponse(request=request, name='status.html',
                                          context={'tab': 'status', 'is_connected': is_printer_connected()})

    @app.get('/api/printer')
    def printer_telemetry():
        ctrl = get_mqtt()
        if ctrl is not None and hasattr(ctrl, "get_telemetry_snapshot"):
            snap = ctrl.get_telemetry_snapshot()
            return JSONResponse(snap, headers={'Cache-Control': 'no-store'})
        return JSONResponse({
            'online': False,
            'uptime_seconds': 0,
            'uptime_formatted': '0分',
            'message': '打印机未连接或 MQTT 控制器离线',
            'gcode_state': 'OFFLINE',
            'temperatures': {'nozzle': 0, 'nozzle_target': 0, 'bed': 0, 'bed_target': 0, 'chamber': 0},
            'fans': {'cooling_fan_speed': 0, 'heatbreak_fan_speed': 0, 'aux_fan_speed': 0},
            'progress_percent': 0,
            'subtask_name': '未连接',
            'stage_text': '离线'
        }, headers={'Cache-Control': 'no-store'})

    @app.get('/login')
    def login_page(request: Request, reason: str = Query(None)):
        cfg = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = {}
        printer_cfg = cfg.get("printer", {})
        cloud_cfg = cfg.get("bambu_cloud", {})
        return templates.TemplateResponse(
            request=request, name='login.html',
            context={
                'tab': 'login',
                'reason': reason,
                'printer': printer_cfg,
                'cloud': cloud_cfg,
                'is_connected': is_printer_connected()
            }
        )

    @app.post('/api/auth/lan-login')
    async def lan_login(request: Request):
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "无效的 JSON 请求体")
        ip = str(body.get("ip", "")).strip()
        access_code = str(body.get("access_code", "")).strip()
        serial = str(body.get("serial", "03900D6A1521014")).strip()

        if not ip:
            raise HTTPException(400, "打印机 IP 不能为空")
        if not access_code:
            raise HTTPException(400, "8位 Access Code 不能为空")
        if not serial.startswith("039"):
            raise HTTPException(400, f"安全约束拦截：仅允许连接拓竹 A1 打印机（序列号前缀必须为 039）。当前输入：{serial}")

        from api.bambu_auth import update_config_printer
        try:
            update_config_printer(config_path, ip=ip, access_code=access_code, target_serial=serial)
        except Exception as e:
            raise HTTPException(500, f"保存配置失败: {e}")

        if on_reconnect is not None:
            try:
                on_reconnect(ip=ip, access_code=access_code, serial=serial)
            except Exception as e:
                return {"status": "ok", "message": f"配置已保存，重连触发异常: {e}", "redirect": "/status"}

        return {"status": "ok", "message": "局域网连接配置已更新并应用", "redirect": "/status"}

    @app.post('/api/auth/cloud-login')
    async def cloud_login(request: Request):
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "无效的 JSON 请求体")
        account = str(body.get("account", "")).strip()
        password = str(body.get("password", "")).strip()
        region = str(body.get("region", "china")).strip().lower()

        if not account or not password:
            raise HTTPException(400, "账号和密码不能为空")

        from api.bambu_auth import BambuAuth
        auth = BambuAuth(region=region)
        res = auth.login_step1(account, password)
        if res.get("status") == "success":
            # Auto fetch device
            try:
                creds = auth.get_a1_credentials()
                from api.bambu_auth import update_config_printer
                update_config_printer(config_path, ip=creds.get("ip", "192.168.199.231"),
                                      access_code=creds["access_code"], target_serial=creds["dev_id"])
                if on_reconnect:
                    on_reconnect(ip=creds.get("ip", "192.168.199.231"),
                                 access_code=creds["access_code"], serial=creds["dev_id"])
                return {"status": "ok", "message": "云端认证并绑定 A1 成功！", "redirect": "/status"}
            except Exception as exc:
                return {"status": "error", "message": f"云端登录成功，但获取 A1 设备信息失败: {exc}"}

        return res

    @app.post('/api/auth/cloud-2fa')
    async def cloud_2fa(request: Request):
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "无效的 JSON 请求体")
        account = str(body.get("account", "")).strip()
        code = str(body.get("code", "")).strip()
        region = str(body.get("region", "china")).strip().lower()

        if not account or not code:
            raise HTTPException(400, "账号和验证码不能为空")

        from api.bambu_auth import BambuAuth
        auth = BambuAuth(region=region)
        res = auth.login_step2(account, code)
        if res.get("status") != "success":
            return res

        try:
            creds = auth.get_a1_credentials()
            from api.bambu_auth import update_config_printer
            update_config_printer(config_path, ip=creds.get("ip", "192.168.199.231"),
                                  access_code=creds["access_code"], target_serial=creds["dev_id"])
            if on_reconnect:
                on_reconnect(ip=creds.get("ip", "192.168.199.231"),
                             access_code=creds["access_code"], serial=creds["dev_id"])
            return {"status": "ok", "message": "两步验证成功，已绑定 A1 打印机！", "redirect": "/status"}
        except Exception as exc:
            return {"status": "error", "message": f"两步验证成功，但提取 A1 访问码失败: {exc}"}

    @app.get('/photos')
    def photos(request: Request, page: int = Query(1, ge=1), task: str = Query(None)):
        try:
            data = store.list_page(page=page, task=task)
        except FileNotFoundError:
            raise HTTPException(404, '照片分类或分页不存在')
        except OSError:
            return templates.TemplateResponse(
                request=request, name='photos.html',
                context={'tab': 'photos', 'photos': [], 'error': '照片目录暂不可用',
                         'page': 1, 'pages': 1, 'total': 0, 'task': 'default', 'tasks': [],
                         'is_connected': is_printer_connected()},
                status_code=503
            )
        return templates.TemplateResponse(request=request, name='photos.html',
                                          context={'tab': 'photos', 'is_connected': is_printer_connected(), **data})

    @app.get('/photos/{filename}')
    def detail(request: Request, filename: str, task: str = Query(None)):
        try:
            data = store.detail(filename, task=task)
        except OSError:
            raise HTTPException(404, '照片不存在')
        return templates.TemplateResponse(request=request, name='detail.html',
                                          context={'tab': 'photos', 'is_connected': is_printer_connected(), **data})

    @app.get('/photos/{task}/{filename}')
    def detail_categorized(request: Request, task: str, filename: str):
        try:
            data = store.detail(filename, task=task)
        except OSError:
            raise HTTPException(404, '照片不存在')
        return templates.TemplateResponse(request=request, name='detail.html',
                                          context={'tab': 'photos', 'is_connected': is_printer_connected(), **data})

    @app.post('/api/photos/capture', status_code=201)
    def take_photo(request: Request):
        from api.frame_recording import CaptureBusy, CaptureUnavailable
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
        return FileResponse(path, headers={'X-Content-Type-Options': 'nosniff'})

    @app.get('/media/{task}/{filename}')
    def image_categorized(task: str, filename: str):
        try:
            path = store.open_image(filename, task=task)
        except OSError:
            raise HTTPException(404, '照片不存在')
        return FileResponse(path, headers={'X-Content-Type-Options': 'nosniff'})

    @app.get('/jetson')
    def jetson(request: Request):
        return templates.TemplateResponse(request=request, name='jetson.html',
                                          context={'tab': 'jetson', 'is_connected': is_printer_connected()})

    @app.get('/api/jetson')
    def metrics():
        return JSONResponse(sampler.snapshot(), headers={'Cache-Control': 'no-store'})

    return app

