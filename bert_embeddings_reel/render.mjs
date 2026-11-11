// Puppeteer (headless Chrome) -> draw(t) per frame -> PNG piped into FFmpeg (H.264).
//
//   node render.mjs                 inject timeline, build audio, render every frame, mux final_reel.mp4
//   node render.mjs --inject        only embed alignment.json into index.html (so file:// preview works)
//   node render.mjs --audio         only build output/audio_final.m4a (also used by the live preview)
//   node render.mjs --stills 5 30   write preview PNGs to output/stills/
//
// The duration is not hardcoded. It is read from the timeline in alignment.json
// (narration length + lead-in + inserted pauses + tail), which the page also uses (window.TOTAL).
import puppeteer from 'puppeteer-core';
import { spawn, spawnSync, execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const FPS = 30, W = 1080, H = 1920, MAX = 120;
const ROOT = path.dirname(new URL(import.meta.url).pathname);
const OUT = path.join(ROOT, 'output');
const CHROME = process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const args = process.argv.slice(2);
fs.mkdirSync(OUT, { recursive: true });
const timeline = () => JSON.parse(fs.readFileSync(path.join(ROOT, 'alignment.json'), 'utf8'));
const run = (cmd, a) => execFileSync(cmd, a, { stdio: 'inherit' });

function inject() {
  const tl = timeline();
  delete tl.raw_response_alignment;              // the page needs only the derived timeline
  const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
  const block = `<!-- DATA:BEGIN (injected by \`node render.mjs --inject\` from alignment.json) -->\n` +
    `<script id="data">window.TIMING = ${JSON.stringify(tl)};</script>\n<!-- DATA:END -->`;
  fs.writeFileSync(path.join(ROOT, 'index.html'), html.replace(/<!-- DATA:BEGIN[\s\S]*?<!-- DATA:END -->/, () => block));
  console.log('injected alignment.json timeline into index.html');
}

function buildAudio() {
  // The animation uses the same warp. Cut the take at each insert point and add silence there,
  // then add the 1 s lead-in and the silent tail. Normalise to -16 LUFS integrated / -1.5 dBTP (two-pass).
  const tm = timeline(), total = tm.warp.total;
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
  const pre = `[cat]adelay=${Math.round(tm.warp.lead_in * 1000)}:all=1,apad,atrim=0:${total}`;
  const graph = l => parts.concat(`${pre},${l}`).join(';');
  // pass 1: measure loudness
  let m;
  try {
    const r = spawnSyncStderr(['-hide_banner', '-i', path.join(ROOT, 'narration.mp3'), '-filter_complex',
      graph('loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json[out]'), '-map', '[out]', '-f', 'null', '-']);
    m = JSON.parse(r.slice(r.lastIndexOf('{'), r.lastIndexOf('}') + 1));
  } catch (e) { m = null; }
  const ln = m ? `loudnorm=I=-16:TP=-1.5:LRA=11:measured_I=${m.input_i}:measured_TP=${m.input_tp}:measured_LRA=${m.input_lra}:` +
    `measured_thresh=${m.input_thresh}:offset=${m.target_offset}:linear=true` : 'loudnorm=I=-16:TP=-1.5:LRA=11';
  run('ffmpeg', ['-y', '-loglevel', 'error', '-i', path.join(ROOT, 'narration.mp3'), '-filter_complex',
    graph(`${ln},aresample=48000,aformat=channel_layouts=stereo,apad,atrim=0:${total}[out]`),
    '-map', '[out]', '-c:a', 'aac', '-b:a', '192k', '-t', String(total), path.join(OUT, 'audio_final.m4a')]);
  console.log('audio: output/audio_final.m4a', m ? `(measured ${m.input_i} LUFS -> -16)` : '');
}
function spawnSyncStderr(a) {   // ffmpeg prints the loudnorm JSON on stderr
  const r = spawnSync('ffmpeg', a, { encoding: 'utf8', maxBuffer: 1 << 26 });
  return r.stderr || '';
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
  const total = await page.evaluate(() => window.TOTAL);           // from the narration timeline
  if (!(total > 0) || total > MAX) throw new Error(`total ${total}s invalid or exceeds ${MAX}s`);
  const n = Math.round(total * FPS);
  const silent = path.join(OUT, 'silent_video.mp4');
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-c:v', 'png', '-framerate', String(FPS), '-i', '-',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-crf', '18', '-preset', 'slow',
    '-r', String(FPS), '-fps_mode', 'cfr', '-s', `${W}x${H}`, silent], { stdio: ['pipe', 'inherit', 'inherit'] });
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
