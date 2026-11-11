// Puppeteer (headless Chrome) -> draw(t) per frame -> PNG frames piped to FFmpeg (H.264) -> mux AAC.
//
//   node render.mjs                  inject timings, build audio, render all frames, write output/deepseek_sparse_attention.mp4
//   node render.mjs --inject         embed timings.json into index.html (so the file:// preview works offline)
//   node render.mjs --audio          only build output/audio_final.m4a (also used by the live preview)
//   node render.mjs --stills 5 30    write PNG stills of given timestamps to output/stills/
//   node render.mjs --workers 6      parallel Chrome pages (default 6); frames are still emitted in order
//
// Env: CHROME_PATH (Chrome/Chromium executable), MUSIC=path/to/bed.mp3 (optional music bed, mixed at -28 dB, ducked).
import puppeteer from 'puppeteer-core';
import { spawn, spawnSync, execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const FPS = 30, W = 1080, H = 1920, MAX = 180;
const ROOT = path.dirname(new URL(import.meta.url).pathname);
const OUT = path.join(ROOT, 'output');
const FINAL = path.join(OUT, 'deepseek_sparse_attention.mp4');
const CHROME = process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d; };
fs.mkdirSync(OUT, { recursive: true });
const timings = () => JSON.parse(fs.readFileSync(path.join(ROOT, 'timings.json'), 'utf8'));
const run = (cmd, a) => execFileSync(cmd, a, { stdio: 'inherit' });

function inject() {
  const tm = timings();
  delete tm.qa; delete tm.pauses;                      // the page only needs timeline + sentences/words
  const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
  const block = `<!-- DATA:BEGIN (injected by \`node render.mjs --inject\` from timings.json) -->\n` +
    `<script id="data">window.TIMING = ${JSON.stringify(tm)};</script>\n<!-- DATA:END -->`;
  fs.writeFileSync(path.join(ROOT, 'index.html'), html.replace(/<!-- DATA:BEGIN[\s\S]*?<!-- DATA:END -->/, () => block));
  console.log('injected timings.json into index.html');
}

function buildAudio() {
  // narration.wav -> 1 s lead-in -> two-pass loudnorm (-14 LUFS, -1.5 dBTP, social target) -> fades
  // -> pad to the animation length (lead-in + narration + final hold) -> AAC 48 kHz stereo.
  const tl = timings().timeline, total = tl.total;
  const nar = path.join(ROOT, 'narration.wav');
  const pre = `[0:a]adelay=${Math.round(tl.lead_in * 1000)}:all=1,apad,atrim=0:${total}`;
  const r = spawnSync('ffmpeg', ['-hide_banner', '-i', nar, '-filter_complex', `${pre},loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json[o]`,
    '-map', '[o]', '-f', 'null', '-'], { encoding: 'utf8', maxBuffer: 1 << 26 });
  const e = r.stderr || '';
  const m = JSON.parse(e.slice(e.lastIndexOf('{'), e.lastIndexOf('}') + 1));
  const ln = `loudnorm=I=-14:TP=-1.5:LRA=11:measured_I=${m.input_i}:measured_TP=${m.input_tp}:measured_LRA=${m.input_lra}:` +
    `measured_thresh=${m.input_thresh}:offset=${m.target_offset}:linear=true`;
  const fades = `afade=t=in:st=0:d=0.4,afade=t=out:st=${(total - 1.2).toFixed(3)}:d=1.2`;
  const inputs = ['-i', nar];
  let graph = `${pre},${ln},aresample=48000,aformat=channel_layouts=stereo,apad,atrim=0:${total}`;
  if (process.env.MUSIC) {                                // optional subtle bed, sidechain-ducked under the voice
    inputs.push('-stream_loop', '-1', '-i', process.env.MUSIC);
    graph += `,asplit=2[v][sc];[1:a]aresample=48000,aformat=channel_layouts=stereo,volume=-28dB,atrim=0:${total}[mus];` +
      `[mus][sc]sidechaincompress=threshold=0.02:ratio=6:attack=20:release=400[duck];[v][duck]amix=inputs=2:duration=first:normalize=0,${fades}[o]`;
  } else graph += `,${fades}[o]`;
  run('ffmpeg', ['-y', '-loglevel', 'error', ...inputs, '-filter_complex', graph, '-map', '[o]',
    '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-t', String(total), path.join(OUT, 'audio_final.m4a')]);
  console.log(`audio: output/audio_final.m4a  (${total}s, measured ${m.input_i} LUFS -> -14)`);
}

async function openPages(n) {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new',
    args: ['--force-device-scale-factor=1', '--allow-file-access-from-files', '--disable-gpu'] });
  const pages = [];
  for (let i = 0; i < n; i++) {
    const page = await browser.newPage();
    page.on('pageerror', e => console.error('page error:', e.message));
    page.on('console', m => { if (m.type() === 'error') console.error('page:', m.text()); });
    await page.setViewport({ width: W, height: H, deviceScaleFactor: 1 });
    await page.evaluateOnNewDocument(() => { window.__RENDERING = true; });
    await page.goto('file://' + path.join(ROOT, 'index.html'), { waitUntil: 'load' });
    await page.evaluate(() => window.ready);
    pages.push(page);
  }
  return { browser, pages };
}
const grab = (page, t) => page.evaluate(t => { draw(t); return document.getElementById('c').toDataURL('image/png').split(',')[1]; }, t)
  .then(b => Buffer.from(b, 'base64'));

async function main() {
  if (argv[0] === '--inject') return inject();
  if (argv[0] === '--audio') return buildAudio();
  inject();
  if (argv[0] === '--stills') {
    const { browser, pages } = await openPages(1);
    fs.mkdirSync(path.join(OUT, 'stills'), { recursive: true });
    for (const t of argv.slice(1).map(Number)) fs.writeFileSync(path.join(OUT, 'stills', `t${t.toFixed(2).padStart(6, '0')}.png`), await grab(pages[0], t));
    return browser.close();
  }
  buildAudio();
  const nw = Number(opt('--workers', 6));
  const { browser, pages } = await openPages(nw);
  const total = await pages[0].evaluate(() => window.TOTAL);
  if (!(total > 0) || total > MAX) throw new Error(`duration ${total}s invalid or exceeds ${MAX}s`);
  const n = Math.round(total * FPS);
  const silent = path.join(OUT, 'video_silent.mp4');
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-c:v', 'png', '-framerate', String(FPS), '-i', '-',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-crf', '18', '-preset', 'medium',
    '-r', String(FPS), '-fps_mode', 'cfr', '-movflags', '+faststart', silent], { stdio: ['pipe', 'inherit', 'inherit'] });
  const t0 = Date.now();
  // Each worker renders frames i ≡ w (mod nw); a reorder buffer writes them to FFmpeg strictly in order.
  const done = new Map(); let next = 0;
  const flush = async () => {
    while (done.has(next)) {
      const b = done.get(next); done.delete(next);
      if (!ff.stdin.write(b)) await new Promise(r => ff.stdin.once('drain', r));
      if (next % 300 === 0) console.log(`frame ${next}/${n}  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
      next++;
    }
  };
  let flushing = Promise.resolve();
  await Promise.all(pages.map(async (page, w) => {
    for (let i = w; i < n; i += nw) {
      while (i - next > nw * 8) await new Promise(r => setTimeout(r, 5));   // bound memory
      done.set(i, await grab(page, i / FPS));
      flushing = flushing.then(flush);
    }
  }));
  await flushing; await flush();
  ff.stdin.end();
  await new Promise(r => ff.on('close', r));
  await browser.close();
  run('ffmpeg', ['-y', '-loglevel', 'error', '-i', silent, '-i', path.join(OUT, 'audio_final.m4a'),
    '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'copy', '-t', String(total), '-movflags', '+faststart', FINAL]);
  console.log(`done: ${FINAL}  (${n} frames, ${total}s, ${((Date.now() - t0) / 1000).toFixed(0)}s render)`);
}
main().catch(e => { console.error(e); process.exit(1); });
