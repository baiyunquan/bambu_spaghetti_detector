import os
from pathlib import Path
import sys
import time

import pytest

ROOT=Path(__file__).resolve().parents[1]
LINE='RAM 1670/3960MB (lfb 162x4MB) SWAP 23/1980MB (cached 2MB) CPU [20%@1479,off,40%@1479,0%@1479] EMC_FREQ 0% GR3D_FREQ 0% PLL@25.5C CPU@27.5C PMIC@50C GPU@26C AO@37C thermal@27.25C'


def module():
    assert (ROOT/'web/telemetry.py').exists(), 'Telemetry not implemented'
    from web import telemetry
    return telemetry


def test_nano_output_online_cpu_and_gpu_without_frequency():
    parsed=module().parse_tegrastats(LINE)
    assert parsed['cpu_percent']==20
    assert parsed['gpu_percent']==0
    assert parsed['ram_used_mb']==1670 and parsed['ram_total_mb']==3960
    assert parsed['swap_used_mb']==23 and parsed['swap_total_mb']==1980
    assert parsed['temperatures']=={'PLL':25.5,'CPU':27.5,'PMIC':50.,'GPU':26.,'AO':37.,'thermal':27.25}


def test_missing_fields_are_null_and_not_previous_values():
    mod=module()
    assert mod.parse_tegrastats('permission denied') is None
    parsed=mod.parse_tegrastats('CPU [off,off] GR3D_FREQ 21%@921 CPU@31.25C')
    assert parsed['cpu_percent'] is None and parsed['gpu_percent']==21
    assert parsed['ram_used_mb'] is None and parsed['swap_used_mb'] is None
    assert mod.parse_tegrastats('RAM 10/3960MB')['gpu_percent'] is None


def test_history_bound_expiry_and_stale_use_monotonic():
    mod=module();now=[0.]
    sampler=mod.TelemetrySampler(clock=lambda:now[0])
    assert sampler.snapshot()['current'] is None
    for i in range(500):
        now[0]=i*2.
        sampler.ingest(LINE)
    result=sampler.snapshot()
    assert result['status']=='ok' and len(result['samples'])==450
    assert result['samples'][0]['age_s']==898
    now[0]+=6.01
    result=sampler.snapshot()
    assert result['status']=='stale' and result['current'] is None
    assert sampler.ingest('unrecognized output') is False
    assert sampler.snapshot()['status']=='stale'
    now[0]+=901
    assert sampler.snapshot()['samples']==[]


def wait_for(predicate,timeout=3):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if predicate():return
        time.sleep(.01)
    raise AssertionError('Timed out')


def test_one_owned_child_reused_and_reaped_even_while_stdout_blocked():
    mod=module()
    command=[sys.executable,'-u','-c','import time;print({!r},flush=True);time.sleep(60)'.format(LINE)]
    sampler=mod.TelemetrySampler(command=command)
    sampler.start();sampler.start()
    try:
        wait_for(lambda:sampler.snapshot()['status']=='ok')
        pid=sampler.pid
        for _ in range(20):assert sampler.snapshot()['current']['gpu_percent']==0
        sampler.start()
        assert sampler.pid==pid
    finally:
        start=time.monotonic();sampler.stop();sampler.stop()
    assert time.monotonic()-start<3
    assert sampler.pid is None
    with pytest.raises(ProcessLookupError):os.kill(pid,0)
    assert sampler.snapshot()['current'] is None


def test_missing_command_retries_with_backoff_without_crashing():
    mod=module();attempts=[]
    def spawn(*args,**kwargs):
        attempts.append(time.monotonic());raise FileNotFoundError('tegrastats missing')
    sampler=mod.TelemetrySampler(popen=spawn,retry_seconds=.05)
    sampler.start()
    try:
        wait_for(lambda:len(attempts)>=3)
        assert sampler.snapshot()['status']=='unavailable'
        assert all(b-a>=.045 for a,b in zip(attempts,attempts[1:]))
    finally:sampler.stop()


def test_multiple_http_requests_share_one_sampler_process(tmp_path):
    mod=module()
    from fastapi.testclient import TestClient
    from web.app import create_app
    command=[sys.executable,'-u','-c','import time;print({!r},flush=True);time.sleep(60)'.format(LINE)]
    sampler=mod.TelemetrySampler(command=command)
    with TestClient(create_app(tmp_path,sampler=sampler)) as browser:
        wait_for(lambda:sampler.snapshot()['status']=='ok')
        pid=sampler.pid
        for _ in range(5):
            assert browser.get('/api/jetson').json()['current']['gpu_percent']==0
            assert browser.get('/jetson').status_code==200
            assert sampler.pid==pid
    assert sampler.pid is None
