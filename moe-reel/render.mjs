import { chromium } from '/opt/homebrew/lib/node_modules/playwright/index.mjs';
import fs from 'fs'; import { spawn } from 'child_process';
const TL = JSON.parse(fs.readFileSync('timeline.json'));
const mode = process.argv[2] || 'stills';
const browser = await chromium.launch({ channel: 'chrome' });
async function mkPage() {
  const p = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
  await p.addInitScript(`window.TIMELINE=${JSON.stringify(TL)}`);
  await p.goto('file://' + process.cwd() + '/reel.html'); await p.waitForFunction(() => window.render);
  return p;
}
async function frame(p, t, type = 'jpeg') {
  const b64 = await p.evaluate(([t, type]) => { window.render(t); return document.getElementById('c').toDataURL(type === 'png' ? 'image/png' : 'image/jpeg', 0.93).split(',')[1]; }, [t, type]);
  return Buffer.from(b64, 'base64');
}
if (mode === 'stills') {
  const p = await mkPage(); fs.mkdirSync('stills', { recursive: true });
  for (const t of process.argv.slice(3).map(Number)) fs.writeFileSync(`stills/t${t}.png`, await frame(p, t, 'png'));
} else {
  const FPS = 30, N = Math.ceil(TL.total * FPS), W = 6;
  fs.mkdirSync('chunks', { recursive: true });
  const per = Math.ceil(N / W);
  await Promise.all([...Array(W).keys()].map(async w => {
    const p = await mkPage(); const a = w * per, b = Math.min(N, a + per);
    const ff = spawn('ffmpeg', ['-y', '-v', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'mjpeg', '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', `chunks/c${w}.mp4`]);
    for (let i = a; i < b; i++) { const buf = await frame(p, i / FPS); if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r)); }
    ff.stdin.end(); await new Promise(r => ff.on('close', r));
  }));
  fs.writeFileSync('chunks/list.txt', [...Array(W).keys()].map(w => `file 'c${w}.mp4'`).join('\n'));
}
await browser.close();
