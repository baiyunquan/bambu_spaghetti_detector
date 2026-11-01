(function () {
  'use strict';
  function points(samples, select, stale) {
    const result = [];
    let previousAge = null;
    for (const sample of samples) {
      if (previousAge !== null && previousAge - sample.age_s > 6) {
        result.push({x: -(previousAge + sample.age_s) / 2, y: null});
      }
      result.push({x: -sample.age_s, y: select(sample) ?? null});
      previousAge = sample.age_s;
    }
    if (stale) result.push({x: 0, y: null});
    return result;
  }
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {points};
    return;
  }
  const colors = ['#2458a6', '#b6532a', '#27815d', '#8655a3', '#8a741a', '#53697e', '#a83e69'];
  const text = (id, value) => {document.getElementById(id).textContent = value;};
  const number = value => Number.isFinite(value) ? value.toFixed(1) : '—';
  function chart(id, temperature) {
    return new Chart(document.getElementById(id), {
      type: 'line', data: {datasets: []}, options: {
        animation: false, responsive: true, maintainAspectRatio: false,
        parsing: false, normalized: true, spanGaps: false,
        elements: {point: {radius: 0}, line: {borderWidth: 1.5, tension: 0}},
        scales: {
          x: {type: 'linear', min: -900, max: 0,
              ticks: {maxTicksLimit: 7, callback: value => value === 0 ? '现在' : `${Math.round(-value / 60)} 分前`}},
          y: {min: temperature ? undefined : 0, max: temperature ? undefined : 100,
              title: {display: true, text: temperature ? '°C' : '%'}}
        },
        plugins: {legend: {position: 'bottom', labels: {boxWidth: 12, boxHeight: 8}},
                  tooltip: {callbacks: {title: items => items.length ? `${Math.round(-items[0].parsed.x)} 秒前` : ''}}}
      }
    });
  }
  const loadChart = chart('load-chart', false);
  const temperatureChart = chart('temperature-chart', true);
  function updateChart(target, definitions, samples, stale) {
    target.data.datasets = definitions.map(([label, select], index) => ({
      label, data: points(samples, select, stale), borderColor: colors[index % colors.length],
      backgroundColor: colors[index % colors.length], spanGaps: false
    }));
    target.update('none');
  }
  function unavailable(message) {
    text('metrics-status', message);
    for (const id of ['cpu-value', 'gpu-value', 'ram-value', 'temperature-value']) text(id, '—');
    text('swap-value', 'SWAP —');
  }
  let timer;
  let inFlight = false;
  async function refresh() {
    if (inFlight || document.hidden) return;
    inFlight = true;
    try {
      const response = await fetch('/api/jetson', {cache: 'no-store', signal: AbortSignal.timeout(5000)});
      if (!response.ok) throw new Error('采集不可用');
      const data = await response.json();
      const samples = data.samples || [];
      const current = data.status === 'ok' ? data.current : null;
      if (current) {
        text('metrics-status', '采集正常');
        text('cpu-value', `${number(current.cpu_percent)} %`);
        text('gpu-value', `${number(current.gpu_percent)} %`);
        text('ram-value', current.ram_used_mb == null ? '—' : `${current.ram_used_mb} / ${current.ram_total_mb} MB`);
        text('swap-value', current.swap_used_mb == null ? 'SWAP —' : `SWAP ${current.swap_used_mb} / ${current.swap_total_mb} MB`);
        const temps = current.temperatures || {};
        text('temperature-value', `${number(temps.CPU)} / ${number(temps.GPU)} °C`);
      } else unavailable(data.message || '采集不可用，等待更新');
      updateChart(loadChart, [
        ['CPU', s => s.cpu_percent], ['GPU', s => s.gpu_percent],
        ['RAM', s => s.ram_total_mb > 0 ? s.ram_used_mb / s.ram_total_mb * 100 : null]
      ], samples, !current);
      const sensors = [...new Set(samples.flatMap(s => Object.keys(s.temperatures || {})))].sort();
      updateChart(temperatureChart, sensors.map(name => [name, s => s.temperatures[name]]), samples, !current);
    } catch (_) {
      unavailable('连接中断，等待重试');
      // Do not leave the last measurement looking current after a network error.
      for (const target of [loadChart, temperatureChart]) {
        target.data.datasets = [];
        target.update('none');
      }
    } finally {
      inFlight = false;
      clearTimeout(timer);
      if (!document.hidden) timer = setTimeout(refresh, 2000);
    }
  }
  document.addEventListener('visibilitychange', () => {
    clearTimeout(timer);
    if (!document.hidden) refresh();
  });
  refresh();
}());
