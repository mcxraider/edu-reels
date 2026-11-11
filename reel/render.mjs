import puppeteer from 'puppeteer-core';
import { spawn } from 'node:child_process';
import path from 'node:path';

const FPS = 30;
const stills = process.argv.slice(2).map(Number);
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: 'new', args: ['--force-device-scale-factor=1']
});
const page = await browser.newPage();
await page.setViewport({ width: 1080, height: 1920 });
await page.goto('file://' + path.resolve('index.html'));
const grab = async t => {
  const b64 = await page.evaluate(t => { draw(t); return document.getElementById('c').toDataURL('image/png').split(',')[1]; }, t);
  return Buffer.from(b64, 'base64');
};
if (stills.length) {
  const fs = await import('node:fs');
  fs.mkdirSync('stills', { recursive: true });
  for (const t of stills) fs.writeFileSync(`stills/t${t}.png`, await grab(t));
} else {
  const total = await page.evaluate(() => TOTAL);
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-i', '-',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'medium', 'video.mp4'], { stdio: ['pipe', 'inherit', 'inherit'] });
  const n = Math.round(total * FPS);
  for (let i = 0; i < n; i++) {
    const buf = await grab(i / FPS);
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
    if (i % 300 === 0) console.log('frame', i, '/', n);
  }
  ff.stdin.end();
  await new Promise(r => ff.on('close', r));
}
await browser.close();
