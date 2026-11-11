// Puppeteer (headless Chrome) -> draw(frame / 30) -> PNG piped straight into FFmpeg (libx264) -> mux voiceover.
//
//   node render.mjs                  inject timings, render every frame, mux output/specinfer_explained.mp4
//   node render.mjs --inject         only embed timings.json into index.html (needed for file:// preview)
//   node render.mjs --stills 5 30.5  write PNG stills to output/stills/
//   node render.mjs --qa             determinism check (renders frames out of order and compares hashes)
//
// Duration is never hardcoded: frames = ceil((measured duration of assets/voiceover.wav + EXTRA_HOLD) * FPS).
// voiceover.wav already contains the 1 s lead-in and the configured ending hold (END_HOLD in generate_audio.py);
// EXTRA_HOLD (env, seconds, default 0) can lengthen the final title card further — the audio is padded with silence.
import puppeteer from 'puppeteer-core';
import { spawn, execFileSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const FPS = 30, W = 1080, H = 1920, MAX_S = 180;
const EXTRA_HOLD = +(process.env.EXTRA_HOLD || 0);
const ROOT = path.dirname(new URL(import.meta.url).pathname);
const OUT = path.join(ROOT, 'output');
const AUDIO = path.join(ROOT, 'assets', 'voiceover.wav');
const FINAL = path.join(OUT, 'specinfer_explained.mp4');
const CHROME = process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const args = process.argv.slice(2);
fs.mkdirSync(OUT, { recursive: true });

const probe = f => JSON.parse(execFileSync('ffprobe', ['-v', 'error', '-show_entries',
  'stream=codec_type,codec_name,width,height,r_frame_rate,pix_fmt,nb_frames:format=duration', '-of', 'json', f]).toString());

function inject(extra = 0) {
  const tm = JSON.parse(fs.readFileSync(path.join(ROOT, 'timings.json'), 'utf8'));
  const slim = { duration: +(tm.duration + extra).toFixed(3), fps: FPS,
    scenes: tm.scenes.map((s, i, a) => ({ ...s, end: i === a.length - 1 ? +(s.end + extra).toFixed(3) : s.end })),
    segments: tm.segments.map(s => ({ id: s.id, scene: s.scene, text: s.text, start: s.start, end: s.end,
      words: s.words.map(w => ({ w: w.w, start: w.start, end: w.end })) })) };
  const file = path.join(ROOT, 'index.html');
  const html = fs.readFileSync(file, 'utf8');
  const block = '<!-- DATA:BEGIN (injected by `node render.mjs --inject` from timings.json) -->\n' +
    `<script id="data">window.TIMING = ${JSON.stringify(slim)};</script>\n<!-- DATA:END -->`;
  fs.writeFileSync(file, html.replace(/<!-- DATA:BEGIN[\s\S]*?<!-- DATA:END -->/, () => block));
  console.log('injected timings.json into index.html');
  return slim;
}

async function openPage() {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new',
    args: ['--force-device-scale-factor=1', '--allow-file-access-from-files', '--font-render-hinting=none'] });
  const page = await browser.newPage();
  page.on('console', m => { if (['error', 'assert', 'warning'].includes(m.type())) console.error('page:', m.text()); });
  page.on('pageerror', e => { console.error('page error:', e.message); process.exitCode = 1; });
  await page.setViewport({ width: W, height: H, deviceScaleFactor: 1 });
  await page.evaluateOnNewDocument(() => { window.__RENDERING = true; });
  await page.goto('file://' + path.join(ROOT, 'index.html'), { waitUntil: 'load' });
  await page.evaluate(() => window.ready);
  const misses = await page.evaluate(() => window.CUE_MISSES);
  if (misses.length) throw new Error('unresolved narration cues: ' + [...new Set(misses)].join(', '));
  return { browser, page };
}
const grab = (page, t) => page.evaluate(t => { draw(t); return document.getElementById('c').toDataURL('image/png').split(',')[1]; }, t)
  .then(b64 => Buffer.from(b64, 'base64'));

async function main() {
  const audioDur = +probe(AUDIO).format.duration;
  const tm = inject(EXTRA_HOLD);
  if (args[0] === '--inject') return;
  if (args[0] === '--stills') {
    const { browser, page } = await openPage();
    fs.mkdirSync(path.join(OUT, 'stills'), { recursive: true });
    for (const t of args.slice(1).map(Number)) fs.writeFileSync(path.join(OUT, 'stills', `t${t.toFixed(2).padStart(6, '0')}.png`), await grab(page, t));
    return browser.close();
  }
  if (args[0] === '--qa') {
    const { browser, page } = await openPage();
    const total = await page.evaluate(() => window.TOTAL);
    const ts = Array.from({ length: 40 }, (_, i) => +(((i * 7919) % 997) / 997 * total).toFixed(3));
    const h = async t => crypto.createHash('md5').update(await grab(page, t)).digest('hex');
    const first = {}; for (const t of ts) first[t] = await h(t);
    let bad = 0; for (const t of [...ts].reverse()) if ((await h(t)) !== first[t]) { bad++; console.error('non-deterministic at', t); }
    console.log(`determinism: ${ts.length} out-of-order timestamps, ${bad} mismatches`);
    await browser.close(); if (bad) process.exit(1); return;
  }
  const total = audioDur + EXTRA_HOLD;
  if (!(total > 0) || total > MAX_S) throw new Error(`duration ${total}s invalid or exceeds ${MAX_S}s`);
  if (Math.abs(tm.duration - total) > 0.1) throw new Error(`timings.json (${tm.duration}s) does not match voiceover.wav (${total}s): re-run generate_audio.py --align-only`);
  const n = Math.ceil(total * FPS - 1e-6);
  console.log(`audio ${audioDur.toFixed(3)} s + extra hold ${EXTRA_HOLD} s -> ${n} frames @ ${FPS} fps`);
  const { browser, page } = await openPage();
  const silent = path.join(OUT, 'video_silent.mp4');
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-c:v', 'png', '-framerate', String(FPS), '-i', '-',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-crf', '18', '-preset', 'slow', '-tune', 'animation',
    '-r', String(FPS), '-fps_mode', 'cfr', '-movflags', '+faststart', silent], { stdio: ['pipe', 'inherit', 'inherit'] });
  const done = new Promise((res, rej) => ff.on('close', c => c ? rej(new Error('ffmpeg exit ' + c)) : res()));
  const t0 = Date.now();
  for (let i = 0; i < n; i++) {
    const buf = await grab(page, i / FPS);
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
    if (i % 300 === 0) console.log(`frame ${i}/${n}  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  ff.stdin.end(); await done; await browser.close();
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', silent, '-i', AUDIO, '-map', '0:v', '-map', '1:a',
    '-c:v', 'copy', '-af', 'apad', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-ac', '2', '-t', (n / FPS).toFixed(3),
    '-movflags', '+faststart', FINAL], { stdio: 'inherit' });
  fs.unlinkSync(silent);
  const p = probe(FINAL), v = p.streams.find(s => s.codec_type === 'video');
  console.log('streams:', p.streams.map(s => `${s.codec_type}:${s.codec_name}`).join(', '),
    `${v.width}x${v.height} ${v.pix_fmt} ${v.r_frame_rate}`, 'duration', p.format.duration);
  if (!p.streams.some(s => s.codec_type === 'audio') || v.width !== W || v.height !== H || v.pix_fmt !== 'yuv420p') throw new Error('bad output streams');
  console.log(`done: ${FINAL} (${n} frames, ${(n / FPS).toFixed(2)} s, ${((Date.now() - t0) / 1000).toFixed(0)} s render)`);
}
main().catch(e => { console.error(e); process.exit(1); });
