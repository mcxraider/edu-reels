// Puppeteer (headless Chrome) -> draw(t) per frame -> PNG -> piped into ffmpeg.
//
//   node render.mjs                 inject timings, build audio, render all frames, mux final_reel.mp4
//   node render.mjs --inject        only embed timings.json into index.html (for file:// preview)
//   node render.mjs --audio         only build output/audio_final.m4a (also used by the live preview)
//   node render.mjs --stills 2 20   write preview PNGs to output/stills/
//
// Duration and frame count come from timings.json (narration + time-warp + thinking period).
import puppeteer from 'puppeteer-core';
import { spawn, execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const FPS = 30, W = 1080, H = 1920, MAX = 90;
const ROOT = path.dirname(new URL(import.meta.url).pathname);
const OUT = path.join(ROOT, 'output');
const CHROME = process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const args = process.argv.slice(2);
fs.mkdirSync(OUT, { recursive: true });
const timing = () => JSON.parse(fs.readFileSync(path.join(ROOT, 'timings.json'), 'utf8'));
const run = (cmd, a) => execFileSync(cmd, a, { stdio: 'inherit' });

function inject() {
  const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
  const block = `<!-- DATA:BEGIN (injected by \`node render.mjs --inject\` from timings.json) -->\n` +
    `<script id="data">window.TIMING = ${JSON.stringify(timing())};</script>\n<!-- DATA:END -->`;
  fs.writeFileSync(path.join(ROOT, 'index.html'), html.replace(/<!-- DATA:BEGIN[\s\S]*?<!-- DATA:END -->/, () => block));
  console.log('injected timings.json into index.html');
}

function buildAudio() {
  // Same warp as the animation: cut the take at each insert point, add silence,
  // 1 s lead-in, pad to total, loudness-normalise for social (-14 LUFS, -1.5 dBTP).
  const tm = timing(), total = tm.warp.total;
  const parts = [], segs = [];
  let prev = 0;
  tm.warp.inserts.forEach((c, i) => {
    parts.push(`[0:a]atrim=${prev}:${c.at},asetpts=PTS-STARTPTS[s${i}]`);
    parts.push(`aevalsrc=0:d=${c.dur}:s=44100:c=mono[z${i}]`);
    segs.push(`[s${i}]`, `[z${i}]`);
    prev = c.at;
  });
  parts.push(`[0:a]atrim=${prev},asetpts=PTS-STARTPTS[slast]`);
  segs.push('[slast]');
  parts.push(`${segs.join('')}concat=n=${segs.length}:v=0:a=1[cat]`);
  parts.push(`[cat]adelay=${Math.round(tm.warp.lead_in * 1000)}:all=1,apad,atrim=0:${total},` +
    `loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000,aformat=channel_layouts=stereo,apad,atrim=0:${total}[out]`);
  run('ffmpeg', ['-y', '-loglevel', 'error', '-i', path.join(ROOT, 'narration.mp3'), '-filter_complex', parts.join(';'),
    '-map', '[out]', '-c:a', 'aac', '-b:a', '192k', '-t', String(total), path.join(OUT, 'audio_final.m4a')]);
  console.log('audio: output/audio_final.m4a');
}

async function openPage() {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new',
    args: ['--force-device-scale-factor=1', '--allow-file-access-from-files'] });
  const page = await browser.newPage();
  page.on('console', m => { if (m.type() === 'error' || m.type() === 'assert') console.error('page:', m.text()); });
  page.on('pageerror', e => console.error('page error:', e.message));
  await page.setViewport({ width: W, height: H, deviceScaleFactor: 1 });
  await page.evaluateOnNewDocument(() => { window.__RENDERING = true; });
  await page.goto('file://' + path.join(ROOT, 'index.html'), { waitUntil: 'load' });
  await page.evaluate(() => window.ready);
  await page.evaluate(() => document.fonts.ready);
  return { browser, page };
}
const grab = (page, t) => page.evaluate(t => { draw(t); return document.getElementById('c').toDataURL('image/png').split(',')[1]; }, t)
  .then(b64 => Buffer.from(b64, 'base64'));

async function main() {
  if (args[0] === '--inject') return inject();
  if (args[0] === '--audio') return buildAudio();
  if (args[0] === '--stills') {
    inject();
    const { browser, page } = await openPage();
    fs.mkdirSync(path.join(OUT, 'stills'), { recursive: true });
    for (const t of args.slice(1).map(Number)) fs.writeFileSync(path.join(OUT, 'stills', `t${t.toFixed(2)}.png`), await grab(page, t));
    await browser.close();
    return;
  }
  inject();
  buildAudio();
  const { browser, page } = await openPage();
  const total = await page.evaluate(() => window.TOTAL);
  if (total > MAX) throw new Error(`total ${total}s exceeds ${MAX}s`);
  const n = Math.round(total * FPS);
  const silent = path.join(OUT, 'silent_video.mp4');
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-c:v', 'png', '-framerate', String(FPS), '-i', '-',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-crf', '17', '-preset', 'slow',
    '-r', String(FPS), '-s', `${W}x${H}`, silent], { stdio: ['pipe', 'inherit', 'inherit'] });
  const t0 = Date.now();
  for (let i = 0; i < n; i++) {
    const buf = await grab(page, i / FPS);
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
    if (i % 300 === 0) console.log(`frame ${i}/${n}  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  ff.stdin.end();
  await new Promise(r => ff.on('close', r));
  await browser.close();
  const final = path.join(ROOT, 'final_reel.mp4');
  run('ffmpeg', ['-y', '-loglevel', 'error', '-i', silent, '-i', path.join(OUT, 'audio_final.m4a'),
    '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'copy', '-t', String(total), '-movflags', '+faststart', final]);
  console.log('done:', final, `(${n} frames, ${total}s)`);
}
main().catch(e => { console.error(e); process.exit(1); });
