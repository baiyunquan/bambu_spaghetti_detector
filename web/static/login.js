(() => {
  const lanTab = document.getElementById('lan-tab');
  const cloudTab = document.getElementById('cloud-tab');
  const lanPane = document.getElementById('lan-pane');
  const cloudPane = document.getElementById('cloud-pane');
  const feedback = document.getElementById('form-feedback');

  function showTab(isLan) {
    if (isLan) {
      lanTab.classList.add('active', 'border-primary', 'border-2');
      lanTab.classList.remove('text-secondary');
      cloudTab.classList.remove('active', 'border-primary', 'border-2');
      cloudTab.classList.add('text-secondary');
      lanPane.classList.add('show', 'active');
      cloudPane.classList.remove('show', 'active');
    } else {
      cloudTab.classList.add('active', 'border-primary', 'border-2');
      cloudTab.classList.remove('text-secondary');
      lanTab.classList.remove('active', 'border-primary', 'border-2');
      lanTab.classList.add('text-secondary');
      cloudPane.classList.add('show', 'active');
      lanPane.classList.remove('show', 'active');
    }
  }

  lanTab.addEventListener('click', () => showTab(true));
  cloudTab.addEventListener('click', () => showTab(false));

  function setFeedback(msg, isError) {
    feedback.textContent = msg;
    feedback.className = `alert ${isError ? 'alert-danger' : 'alert-success'} mb-4`;
    feedback.classList.remove('d-none');
    feedback.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // LAN Form Submit
  const lanForm = document.getElementById('lan-form');
  const lanBtn = document.getElementById('lan-submit-btn');
  lanForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    lanBtn.disabled = true;
    lanBtn.textContent = '连接验证中...';
    feedback.classList.add('d-none');

    const payload = {
      ip: document.getElementById('printer-ip').value.trim(),
      access_code: document.getElementById('access-code').value.trim(),
      serial: document.getElementById('printer-serial').value.trim()
    };

    try {
      const resp = await fetch('/api/auth/lan-login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      if (!resp.ok || data.status === 'error') {
        throw new Error(data.detail || data.message || '连接保存失败');
      }
      setFeedback(data.message + '，正在跳转状态页...', false);
      setTimeout(() => { window.location.href = data.redirect || '/status'; }, 1200);
    } catch (err) {
      setFeedback(err.message, true);
    } finally {
      lanBtn.disabled = false;
      lanBtn.textContent = '保存并连接打印机';
    }
  });

  // Cloud Form Submit
  const cloudForm = document.getElementById('cloud-form');
  const cloudBtn = document.getElementById('cloud-submit-btn');
  const cloud2faSec = document.getElementById('cloud-2fa-section');
  const cloud2faBtn = document.getElementById('cloud-2fa-btn');

  cloudForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    cloudBtn.disabled = true;
    cloudBtn.textContent = '云端认证中...';
    feedback.classList.add('d-none');

    const payload = {
      account: document.getElementById('cloud-account').value.trim(),
      password: document.getElementById('cloud-password').value,
      region: document.getElementById('cloud-region').value
    };

    try {
      const resp = await fetch('/api/auth/cloud-login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || data.message || '登录异常');

      if (data.status === '2fa_required') {
        cloud2faSec.classList.remove('d-none');
        cloudBtn.classList.add('d-none');
        setFeedback(data.message, false);
      } else if (data.status === 'ok') {
        setFeedback(data.message + '，正在跳转状态页...', false);
        setTimeout(() => { window.location.href = data.redirect || '/status'; }, 1200);
      } else {
        throw new Error(data.message || '登录失败');
      }
    } catch (err) {
      setFeedback(err.message, true);
    } finally {
      cloudBtn.disabled = false;
      cloudBtn.textContent = '登录并自动同步 A1 凭证';
    }
  });

  // Cloud 2FA Submit
  cloud2faBtn.addEventListener('click', async () => {
    const code = document.getElementById('cloud-2fa-code').value.trim();
    if (!code) {
      alert('请输入邮箱收到的验证码');
      return;
    }
    cloud2faBtn.disabled = true;
    cloud2faBtn.textContent = '验证中...';

    const payload = {
      account: document.getElementById('cloud-account').value.trim(),
      code: code,
      region: document.getElementById('cloud-region').value
    };

    try {
      const resp = await fetch('/api/auth/cloud-2fa', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      if (!resp.ok || data.status !== 'ok') throw new Error(data.detail || data.message || '2FA 验证失败');
      setFeedback(data.message + '，正在跳转状态页...', false);
      setTimeout(() => { window.location.href = data.redirect || '/status'; }, 1200);
    } catch (err) {
      setFeedback(err.message, true);
    } finally {
      cloud2faBtn.disabled = false;
      cloud2faBtn.textContent = '确认验证并绑定';
    }
  });
})();
