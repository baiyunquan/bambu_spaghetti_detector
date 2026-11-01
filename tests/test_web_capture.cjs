const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function page(fetch) {
  let click;
  const button = {disabled: false, addEventListener: (event, fn) => { click = fn; }};
  const status = {textContent: ''};
  const navigations = [];
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../web/static/capture.js'), 'utf8'), {
    document: {getElementById: id => id === 'capture-button' ? button : status},
    window: {location: {assign: url => navigations.push(url)}}, fetch, AbortSignal
  });
  return {button, status, navigations, click: () => click()};
}

(async () => {
  let finish;
  const pending = new Promise(resolve => { finish = resolve; });
  const success = page(async (url, options) => {
    assert.equal(url, '/api/photos/capture');
    assert.equal(options.method, 'POST');
    assert.equal(options.headers['X-Capture-Request'], '1');
    return pending;
  });
  const result = success.click();
  assert.equal(success.button.disabled, true);
  assert.match(success.status.textContent, /等待新的相机画面/);
  finish({ok: true, json: async () => ({url: '/photos/manual.jpg'})});
  await result;
  assert.deepEqual(success.navigations, ['/photos/manual.jpg']);

  const failure = page(async () => ({ok: false, json: async () => ({detail: '未收到新的相机画面'})}));
  await failure.click();
  assert.equal(failure.button.disabled, false);
  assert.equal(failure.status.textContent, '未收到新的相机画面');
  assert.deepEqual(failure.navigations, []);
  console.log('Capture pending, POST, detail navigation and retry tests passed');
})();
