// Polling printer telemetry from /api/printer every 1.5 seconds
(() => {
  let timer = null;

  const STATE_BADGES = {
    'RUNNING': { text: '正在打印', cls: 'bg-primary' },
    'PAUSE': { text: '已暂停', cls: 'bg-warning text-dark' },
    'IDLE': { text: '空闲就绪', cls: 'bg-success' },
    'FINISH': { text: '打印完成', cls: 'bg-info text-dark' },
    'PREPARE': { text: '准备中', cls: 'bg-secondary' },
    'FAILED': { text: '打印失败', cls: 'bg-danger' },
    'OFFLINE': { text: '离线', cls: 'bg-dark' },
    'UNKNOWN': { text: '未知状态', cls: 'bg-secondary' }
  };

  async function updateTelemetry() {
    try {
      const resp = await fetch('/api/printer', { cache: 'no-store' });
      if (!resp.ok) throw new Error('API failed with ' + resp.status);
      const data = await resp.json();
      renderStatus(data);
    } catch (err) {
      console.warn('Telemetry fetch error:', err);
      renderOffline();
    }
  }

  function renderStatus(data) {
    const isOnline = Boolean(data.online);
    const offlineAlert = document.getElementById('offline-alert');
    const statusBadge = document.getElementById('printer-status-badge');
    const uptimeBadge = document.getElementById('printer-uptime-badge');
    const ipSerial = document.getElementById('printer-ip-serial');

    if (!isOnline) {
      if (offlineAlert) offlineAlert.classList.remove('d-none');
      if (statusBadge) {
        statusBadge.textContent = '离线 / 未连接';
        statusBadge.className = 'badge bg-danger';
      }
      if (uptimeBadge) uptimeBadge.textContent = '在线: 0分';
    } else {
      if (offlineAlert) offlineAlert.classList.add('d-none');
      const st = (data.gcode_state || 'IDLE').toUpperCase();
      const badgeInfo = STATE_BADGES[st] || { text: st, cls: 'bg-primary' };
      if (statusBadge) {
        statusBadge.textContent = badgeInfo.text;
        statusBadge.className = 'badge ' + badgeInfo.cls;
      }
      if (uptimeBadge) uptimeBadge.textContent = '已在线 ' + (data.uptime_formatted || '0分');
    }

    if (ipSerial && data.printer_ip) {
      ipSerial.textContent = `设备: 拓竹 A1 | IP: ${data.printer_ip} | SN: ${data.dev_id || '--'}`;
    }

    // Print Job info
    const taskName = document.getElementById('task-name');
    const gcodeFile = document.getElementById('gcode-file');
    const progressPercent = document.getElementById('progress-percent');
    const progressBar = document.getElementById('progress-bar');
    const currentLayer = document.getElementById('current-layer');
    const totalLayers = document.getElementById('total-layers');
    const remainingTime = document.getElementById('remaining-time');
    const printStage = document.getElementById('print-stage');
    const printSpeed = document.getElementById('print-speed');

    const pct = Number(data.progress_percent || 0);
    if (taskName) taskName.textContent = data.subtask_name || '无活动打印任务';
    if (gcodeFile) gcodeFile.textContent = data.gcode_file || '--';
    if (progressPercent) progressPercent.textContent = pct + '%';
    if (progressBar) {
      progressBar.style.width = pct + '%';
      progressBar.setAttribute('aria-valuenow', pct);
    }
    if (currentLayer) currentLayer.textContent = data.current_layer || 0;
    if (totalLayers) totalLayers.textContent = data.total_layers || 0;
    if (remainingTime) remainingTime.textContent = data.remaining_time_formatted || '--';
    if (printStage) printStage.textContent = data.stage_text || '空闲就绪';
    if (printSpeed) printSpeed.textContent = data.speed_text || '标准 (100%)';

    // Temperatures
    const temps = data.temperatures || {};
    const nozzleCur = document.getElementById('temp-nozzle');
    const nozzleTar = document.getElementById('target-nozzle');
    const bedCur = document.getElementById('temp-bed');
    const bedTar = document.getElementById('target-bed');
    const nozzleHint = document.getElementById('nozzle-state-hint');
    const bedHint = document.getElementById('bed-state-hint');

    if (nozzleCur) nozzleCur.textContent = Number(temps.nozzle || 0).toFixed(1);
    if (nozzleTar) nozzleTar.textContent = Number(temps.nozzle_target || 0).toFixed(0);
    if (bedCur) bedCur.textContent = Number(temps.bed || 0).toFixed(1);
    if (bedTar) bedTar.textContent = Number(temps.bed_target || 0).toFixed(0);

    if (nozzleHint) {
      const tar = Number(temps.nozzle_target || 0);
      const cur = Number(temps.nozzle || 0);
      if (tar > 0 && Math.abs(cur - tar) > 2) {
        nozzleHint.textContent = `目标 ${tar}℃ (加热中...)`;
        nozzleHint.className = 'small text-warning fw-semibold';
      } else {
        nozzleHint.textContent = tar > 0 ? `目标稳定在 ${tar}℃` : '未加热';
        nozzleHint.className = 'small text-secondary';
      }
    }

    if (bedHint) {
      const tar = Number(temps.bed_target || 0);
      const cur = Number(temps.bed || 0);
      if (tar > 0 && Math.abs(cur - tar) > 1.5) {
        bedHint.textContent = `目标 ${tar}℃ (升温中...)`;
        bedHint.className = 'small text-warning fw-semibold';
      } else {
        bedHint.textContent = tar > 0 ? `目标稳定在 ${tar}℃` : '未加热';
        bedHint.className = 'small text-secondary';
      }
    }

    // Fans
    const fans = data.fans || {};
    const fanCooling = document.getElementById('fan-cooling');
    const fanHeatbreak = document.getElementById('fan-heatbreak');
    if (fanCooling) fanCooling.textContent = fans.cooling_fan_speed || 0;
    if (fanHeatbreak) fanHeatbreak.textContent = fans.heatbreak_fan_speed || 0;

    // Filament / Tray
    const tray = data.active_tray || {};
    const trayDot = document.getElementById('tray-color-dot');
    const trayMaterial = document.getElementById('tray-material');
    const trayRemain = document.getElementById('tray-remain');
    const trayName = document.getElementById('tray-name');

    if (trayDot && tray.color) trayDot.style.backgroundColor = tray.color;
    if (trayMaterial) trayMaterial.textContent = tray.type || 'PLA';
    if (trayRemain) trayRemain.textContent = tray.remain_percent !== undefined ? tray.remain_percent : 100;
    if (trayName) trayName.textContent = tray.name || '外部料槽';

    // Hardware
    const hw = data.hardware || {};
    const wifiSig = document.getElementById('wifi-signal');
    const sdcard = document.getElementById('sdcard-status');
    const lastUpdate = document.getElementById('last-update-time');

    if (wifiSig) wifiSig.textContent = hw.wifi_signal || '--';
    if (sdcard) sdcard.textContent = hw.sdcard ? '正常已插入' : '未检测到';
    if (lastUpdate) lastUpdate.textContent = new Date().toLocaleTimeString();
  }

  function renderOffline() {
    const offlineAlert = document.getElementById('offline-alert');
    const statusBadge = document.getElementById('printer-status-badge');
    if (offlineAlert) offlineAlert.classList.remove('d-none');
    if (statusBadge) {
      statusBadge.textContent = '服务响应异常';
      statusBadge.className = 'badge bg-danger';
    }
  }

  // Start polling
  updateTelemetry();
  timer = setInterval(updateTelemetry, 1500);

  // Clean on page hide
  window.addEventListener('beforeunload', () => {
    if (timer) clearInterval(timer);
  });
})();
