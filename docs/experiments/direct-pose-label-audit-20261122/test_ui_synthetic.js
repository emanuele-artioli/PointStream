// Synthetic DOM-only behavior test; no browser, real labels or screenshots.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
const html = fs.readFileSync(process.argv[2], 'utf8');
const script = html.match(/<script>([\s\S]*)<\/script>/)[1];
const elements = {};
const classList = {toggle(){}};
function node(id) { return elements[id] ||= {id, classList, textContent:'', children:[], addEventListener(n,f){this['on'+n]=f}, appendChild(x){this.children.push(x)}, replaceChildren(){this.children=[]}, getContext(){return {drawImage(){}, beginPath(){}, arc(){}, fill(){}, stroke(){}, clearRect(){}, moveTo(){}, lineTo(){}}}, getBoundingClientRect(){return {left:0,top:0,width:960,height:540}}, click(){if(this.onclick)this.onclick()}, blur(){}}; }
const document = {getElementById:node, createElement:()=>node('generated-'+Math.random()), addEventListener(){}};
let saved = {};
const localStorage = {getItem(k){return saved[k]||null},setItem(k,v){saved[k]=v}};
class Image {set src(value){this._src=value;if(this.onload)this.onload()} get complete(){return true}}
vm.runInNewContext(script,{document,localStorage,Image,console,Math,JSON,URL:{createObjectURL(){return 'blob:synthetic'},revokeObjectURL(){}},Blob, setTimeout});
assert.match(node('frame').textContent,/tick 10/);
assert.match(node('progress').textContent,/0 \/ 408/);
node('view').onpointerdown({clientX:100,clientY:200});
let data=JSON.parse(Object.values(saved)[0]);
assert.deepStrictEqual(JSON.parse(JSON.stringify(data[0][0][0])),{status:'visible',x:200,y:400});
node('occluded').click(); data=JSON.parse(Object.values(saved)[0]);
assert.strictEqual(data[0][0][1].status,'occluded');
node('p1').click();node('unsure').click(); data=JSON.parse(Object.values(saved)[0]);
assert.strictEqual(data[0][1][2].status,'unsure');
node('next').click();assert.match(node('frame').textContent,/tick 30/);
node('export').click();assert.match(node('notice').textContent,/Mark every joint/);
console.log('synthetic DOM behavior: source-coordinate mapping, selection, status, frame navigation, incomplete export passed');
