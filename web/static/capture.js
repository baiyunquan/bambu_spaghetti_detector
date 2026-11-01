'use strict';
const captureButton = document.getElementById('capture-button');
const captureStatus = document.getElementById('capture-status');
captureButton.addEventListener('click', async () => {
  captureButton.disabled = true;
  captureStatus.textContent = '等待新的相机画面并保存…';
  try {
    const response = await fetch('/api/photos/capture', {
      method: 'POST', headers: {'X-Capture-Request': '1'}, cache: 'no-store',
      signal: AbortSignal.timeout(15000)
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || '拍摄失败，请稍后重试');
    window.location.assign(result.url);
  } catch (error) {
    captureStatus.textContent = error.name === 'TimeoutError' ? '拍摄超时，请检查相机连接后重试' : (error.message || '连接中断，请重试');
    captureButton.disabled = false;
  }
});
