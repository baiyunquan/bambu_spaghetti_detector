const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const filename = path.resolve(__dirname, '../web/static/jetson.js');
const {points} = require(filename);
assert.deepEqual(points([{age_s:12,value:0},{age_s:2,value:null}], s=>s.value, true),
  [{x:-12,y:0},{x:-7,y:null},{x:-2,y:null},{x:0,y:null}]);
let listener, requests = 0;
const document = {hidden:false, getElementById:()=>({textContent:''}),
                  addEventListener:(_,callback)=>{listener=callback;}};
vm.runInNewContext(fs.readFileSync(filename, 'utf8'), {
  document, Chart:class {constructor(){this.data={datasets:[]};} update(){}},
  fetch:()=>{requests++;return new Promise(()=>{});},
  AbortSignal:{timeout:()=>({})},setTimeout:()=>1,clearTimeout:()=>{}
});
listener();listener();listener();
assert.equal(requests,1,'Visibility changes must not create parallel polling loops');
console.log('Chart gaps, zero/null and single-poll tests passed');
