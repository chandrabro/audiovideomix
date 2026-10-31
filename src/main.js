import { FFmpeg } from '@ffmpeg/ffmpeg';
import { fetchFile, toBlobURL } from '@ffmpeg/util';
import Tesseract from 'tesseract.js';

const $ = id => document.getElementById(id);
const log = m => { $('log').textContent += m + '\n'; $('log').scrollTop = 1e9; };
const esc = s => s.replace(/[&<>"]/g, c => '&#' + c.charCodeAt(0) + ';');
const norm = t => t.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, ' ').replace(/\s+/g, ' ').trim();
let ff, segs = [], vinfo, atotal = 0;

async function getFF() {
  if (ff) return ff;
  ff = new FFmpeg();
  ff.on('log', ({ message }) => { if (/error|invalid/i.test(message)) log(message); });
  const base = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/esm';
  await ff.load({
    coreURL: await toBlobURL(`${base}/ffmpeg-core.js`, 'text/javascript'),
    wasmURL: await toBlobURL(`${base}/ffmpeg-core.wasm`, 'application/wasm'),
  });
  return ff;
}

// ---------- SRT helpers ----------
const ts = t => { const m = t.trim().match(/(\d+):(\d+):(\d+)[,.](\d+)/); return +m[1] * 3600 + +m[2] * 60 + +m[3] + +('0.' + m[4]); };
function parseSrt(txt) {
  return txt.replace(/\r/g, '').trim().split(/\n\s*\n/).map(b => {
    const l = b.split('\n'), i = l.findIndex(x => x.includes('-->'));
    if (i < 0) return null;
    const [a, c] = l[i].split('-->');
    return { s: ts(a), e: ts(c), text: norm(l.slice(i + 1).join(' ').replace(/<[^>]+>/g, ' ')) };
  }).filter(x => x && x.text);
}
const meta = file => new Promise(r => {
  const v = document.createElement('video'); v.preload = 'metadata';
  v.onloadedmetadata = () => r({ dur: v.duration, w: v.videoWidth, h: v.videoHeight });
  v.src = URL.createObjectURL(file);
});
// video SRT -> scenes (each cue starts a scene; scenes are contiguous)
const scenesFromSrt = (k, dur) => k.map((c, i) => ({ vs: i ? c.s : 0, text: c.text, ve: Math.min(k[i + 1] ? k[i + 1].s : dur, dur) }));
// audio SRT -> words with estimated times (spread evenly across each cue)
function wordsFromSrt(k) {
  const words = [];
  k.forEach(c => { const ws = c.text.split(' '), d = (c.e - c.s) / ws.length; ws.forEach((w, i) => words.push({ w, s: c.s + i * d })); });
  return words;
}

// ---------- 1. OCR the captions ----------
const seek = (v, t) => new Promise(r => { v.onseeked = () => r(); v.currentTime = Math.max(t, 0.001); });

async function readCaptions(file, step) {
  const v = document.createElement('video'); v.muted = true; v.src = URL.createObjectURL(file);
  await new Promise(r => v.onloadedmetadata = r);
  const c = document.createElement('canvas'); c.width = v.videoWidth; c.height = v.videoHeight;
  const x = c.getContext('2d');
  const sm = document.createElement('canvas'); sm.width = sm.height = 32;
  const sx = sm.getContext('2d', { willReadFrequently: true });
  const worker = await Tesseract.createWorker($('lang').value || 'eng');
  const rows = [], dur = v.duration; let prev = null, last = '';
  for (let t = 0; t < dur; t += step) {
    await seek(v, Math.min(t, dur - 0.01));
    sx.drawImage(v, 0, 0, 32, 32);
    const d = sx.getImageData(0, 0, 32, 32).data;
    let diff = prev ? 0 : 1e9;
    if (prev) for (let i = 0; i < d.length; i += 4) diff += Math.abs(d[i] - prev[i]);
    prev = d;
    if (diff / 1024 > 3) { // frame changed -> OCR it
      x.drawImage(v, 0, 0);
      last = norm((await worker.recognize(c)).data.text);
    }
    rows.push({ t, text: last });
    if (rows.length % 10 === 0) log(`OCR ${t.toFixed(1)} / ${dur.toFixed(1)}s`);
  }
  await worker.terminate();
  return { rows, dur, w: v.videoWidth, h: v.videoHeight };
}

const sim = (a, b) => {
  if (!a || !b) return a === b ? 1 : 0;
  const A = a.split(' '), B = new Set(b.split(' '));
  return 2 * A.filter(w => B.has(w)).length / (A.length + B.size);
};

function group(rows, step, dur) {
  const out = [];
  for (const r of rows) {
    const p = out[out.length - 1];
    if (p && (r.text.length < 3 || sim(p.text, r.text) > 0.6)) {
      if (r.text.length > p.text.length && sim(p.text, r.text) > 0.6) p.text = r.text;
    } else out.push({ vs: r.t, text: r.text });
  }
  const k = out.filter(s => s.text.length >= 3);
  if (k.length) k[0].vs = 0;
  k.forEach((s, i) => s.ve = k[i + 1] ? k[i + 1].vs : dur);
  return k;
}

// ---------- 2. Audio -> words with timestamps ----------
async function buildAudio(files, args, ext, out) {
  const f = await getFF(), names = [];
  for (let i = 0; i < files.length; i++) {
    await f.writeFile('a' + i, await fetchFile(files[i]));
    await f.exec(['-i', 'a' + i, '-vn', ...args, `${out}${i}.${ext}`]);
    names.push(`${out}${i}.${ext}`);
  }
  await f.writeFile(out + '.txt', names.map(n => `file '${n}'`).join('\n'));
  await f.exec(['-f', 'concat', '-safe', '0', '-i', out + '.txt', '-c', 'copy', `${out}.${ext}`]);
  return f.readFile(`${out}.${ext}`);
}

async function transcribe(files) {
  const data = await buildAudio(files, ['-ac', '1', '-ar', '16000', '-b:a', '32k'], 'mp3', 'w');
  const fd = new FormData();
  fd.append('file', new Blob([data], { type: 'audio/mpeg' }), 'audio.mp3');
  fd.append('model', 'whisper-1'); fd.append('response_format', 'verbose_json');
  fd.append('timestamp_granularities[]', 'word');
  const r = await fetch('https://api.openai.com/v1/audio/transcriptions',
    { method: 'POST', headers: { Authorization: 'Bearer ' + $('key').value.trim() }, body: fd });
  if (!r.ok) throw new Error(await r.text());
  const j = await r.json();
  return { words: j.words.map(w => ({ w: norm(w.word), s: w.start })).filter(x => x.w), total: j.duration };
}

// ---------- 3. Match each caption scene to its spoken words ----------
function align(segs, words) {
  let p = 0;
  segs.forEach(s => {
    const C = s.text.split(' '), n = C.length, cs = new Set(C);
    let best = -1, bi = p;
    for (let i = p; i < Math.min(words.length, p + 3 * n + 40); i++) {
      let sc = 0;
      for (let j = i; j < Math.min(words.length, i + n); j++) if (cs.has(words[j].w)) sc++;
      if (sc > best) { best = sc; bi = i; }
    }
    s.conf = best / n;
    s.as = words[Math.min(bi, words.length - 1)].s;
    p = Math.min(bi + Math.max(1, Math.round(n * 0.7)), words.length - 1);
  });
  segs[0].as = 0;
  for (let i = 1; i < segs.length; i++) segs[i].as = Math.max(segs[i].as, segs[i - 1].as + 0.2);
}

const aEnd = i => segs[i + 1] ? segs[i + 1].as : atotal;
const lim = () => [+$('minS').value, +$('maxS').value];

function draw() {
  const [mn, mx] = lim();
  const num = (s, k, i) => `<td><input type=number step=0.05 value="${s[k].toFixed(2)}" data-k="${k}" data-i="${i}">`;
  $('tbl').innerHTML = '<tr><th>#<th>Caption (OCR)<th>Video start<th>Video end<th>Audio start<th>Speed<th>Match</tr>' +
    segs.map((s, i) => {
      const f = (s.ve - s.vs) / Math.max(aEnd(i) - s.as, 0.1);
      return `<tr><td>${i + 1}<td>${esc(s.text.slice(0, 60))}${num(s, 'vs', i)}${num(s, 've', i)}${num(s, 'as', i)}` +
        `<td>${f.toFixed(2)}x${f < mn || f > mx ? ' ⚠' : ''}<td>${Math.round(s.conf * 100)}%${s.conf < .5 ? ' ⚠' : ''}</tr>`;
    }).join('');
}
$('tbl').onchange = e => { segs[e.target.dataset.i][e.target.dataset.k] = +e.target.value; draw(); };

$('analyze').onclick = async () => {
  try {
    const vf = $('video').files[0], af = [...$('audio').files], vsrt = $('vsrt').files[0], asrt = $('asrt').files[0];
    if (!vf || !af.length) return alert('Add a video and audio file(s).');
    if (!asrt && !$('key').value) return alert('Add your OpenAI key, or upload an audio SRT instead.');
    $('log').textContent = ''; $('render').disabled = true;
    if (vsrt) {
      log('Using video SRT (skipping OCR)…');
      const m = await meta(vf); vinfo = m;
      segs = scenesFromSrt(parseSrt(await vsrt.text()), m.dur);
    } else {
      const step = 1 / +$('sps').value;
      log('Reading captions (OCR)…');
      vinfo = await readCaptions(vf, step);
      segs = group(vinfo.rows, step, vinfo.dur);
    }
    log(`${segs.length} caption scenes.`);
    let words;
    if (asrt) {
      log('Using audio SRT (skipping transcription)…');
      atotal = 0; for (const f of af) atotal += (await meta(f)).dur;
      words = wordsFromSrt(parseSrt(await asrt.text()));
    } else {
      log('Transcribing audio…');
      const tr = await transcribe(af); atotal = tr.total; words = tr.words;
    }
    if (!segs.length || !words.length) throw new Error('No captions or no spoken words found.');
    align(segs, words); draw();
    $('render').disabled = false; log('Review the table (fix anything marked ⚠), then Render.');
  } catch (e) { log('ERROR: ' + e.message); }
};

// ---------- 4. Render: per-scene speed change, hard cuts ----------
$('render').onclick = async () => {
  try {
    const f = await getFF(), { w, h } = vinfo, [mn, mx] = lim(), mode = $('pad').value, FPS = 30;
    const w2 = w - (w % 2), h2 = h - (h % 2);
    await f.writeFile('in.vid', await fetchFile($('video').files[0]));
    if (mode === 'image' && $('img').files[0]) await f.writeFile('img', await fetchFile($('img').files[0]));
    const list = [];
    for (let i = 0; i < segs.length; i++) {
      const s = segs[i], vd = s.ve - s.vs, td = Math.max(aEnd(i) - s.as, 0.1);
      const sp = Math.min(mx, Math.max(mn, vd / td)), got = vd / sp, gap = td - got;
      let vf = `setpts=PTS/${sp},fps=${FPS},scale=${w2}:${h2},setsar=1`;
      if (gap > 0.04 && mode === 'freeze') vf += `,tpad=stop_mode=clone:stop_duration=${gap}`;
      if (gap > 0.04 && mode === 'black') vf += `,tpad=stop_mode=add:stop_duration=${gap}:color=black`;
      await f.exec(['-ss', String(s.vs), '-t', String(vd), '-i', 'in.vid', '-vf', vf, '-an', '-t', String(td),
        '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-r', String(FPS), `s${i}.mp4`]);
      list.push(`s${i}.mp4`);
      if (gap > 0.04 && mode === 'image' && $('img').files[0]) {
        await f.exec(['-loop', '1', '-i', 'img', '-t', String(gap), '-vf',
          `scale=${w2}:${h2}:force_original_aspect_ratio=decrease,pad=${w2}:${h2}:(ow-iw)/2:(oh-ih)/2,fps=${FPS},setsar=1`,
          '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', `p${i}.mp4`]);
        list.push(`p${i}.mp4`);
      }
      log(`Scene ${i + 1}/${segs.length}: ${sp.toFixed(2)}x${gap > 0.04 ? ', +' + gap.toFixed(1) + 's fill' : ''}`);
    }
    await f.writeFile('v.txt', list.map(n => `file '${n}'`).join('\n'));
    await f.exec(['-f', 'concat', '-safe', '0', '-i', 'v.txt', '-c', 'copy', 'video.mp4']);
    log('Mixing audio…');
    await buildAudio([...$('audio').files], ['-ar', '44100', '-ac', '2', '-b:a', '160k'], 'm4a', 'h');
    await f.exec(['-i', 'video.mp4', '-i', 'h.m4a', '-map', '0:v', '-map', '1:a', '-c', 'copy', '-shortest', 'out.mp4']);
    const url = URL.createObjectURL(new Blob([await f.readFile('out.mp4')], { type: 'video/mp4' }));
    $('out').src = url; $('dl').href = url; log('Done.');
  } catch (e) { log('ERROR: ' + e.message); }
};
