import { FFmpeg } from '@ffmpeg/ffmpeg';
import { fetchFile, toBlobURL } from '@ffmpeg/util';
import Tesseract from 'tesseract.js';

const $ = id => document.getElementById(id);
const log = m => { $('log').textContent += m + '\n'; $('log').scrollTop = 1e9; };
const esc = s => s.replace(/[&<>"]/g, c => '&#' + c.charCodeAt(0) + ';');
const norm = t => t.toLowerCase().replace(/[^\p{L}\p{M}\p{N}\s]/gu, ' ').replace(/\s+/g, ' ').trim();
let ff, segs = [], awords = [], vinfo, atotal = 0, busy = false, ready = false, tail = [];
const fail = e => { log('ERROR: ' + (e && e.message ? e.message : String(e))); console.error(e); if (ff) { try { ff.terminate(); } catch {} ff = null; } };
const run = async (f, a) => { tail = []; const c = await f.exec(a); if (c !== 0) throw new Error(`ffmpeg exit ${c}: ${tail.slice(-4).join(' | ')}`); };

async function probe(f, name) {
  tail = []; await f.exec(['-i', name]); // exits non-zero (no output file) but logs the duration
  const m = tail.join(' ').match(/Duration:\s*(\d+):(\d+):([\d.]+)/);
  return m ? +m[1] * 3600 + +m[2] * 60 + +m[3] : 0;
}

async function mountVideo(file) {
  const f = await getFF();
  try { await f.createDir('/data'); } catch {}
  try { await f.unmount('/data'); } catch {}
  await f.mount('WORKERFS', { files: [file] }, '/data');
  return '/data/' + file.name;
}

async function getFF() {
  if (ff) return ff;
  ff = new FFmpeg();
  ff.on('log', ({ message }) => { tail.push(message); if (/error|invalid/i.test(message)) log(message); });
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
    await run(f, ['-i', 'a' + i, '-vn', ...args, `${out}${i}.${ext}`]);
    names.push(`${out}${i}.${ext}`);
  }
  await f.writeFile(out + '.txt', names.map(n => `file '${n}'`).join('\n'));
  await run(f, ['-f', 'concat', '-safe', '0', '-i', out + '.txt', '-c', 'copy', `${out}.${ext}`]);
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

// ---------- 3. Match scenes to audio: global alignment of the two word sequences ----------
// (tolerates extra sounds like "hahaha", missing words, small spelling differences)
function dpAlign(V, A) {
  const n = V.length, m = A.length, B = Math.max(150, Math.ceil(0.2 * Math.max(n, m))), W = 2 * B + 1;
  const bg = w => { const s = new Set(); for (let i = 0; i < w.length - 1; i++) s.add(w.slice(i, i + 2)); return s; };
  const bv = V.map(bg), ba = A.map(bg);
  const score = (i, j) => {
    if (V[i] === A[j]) return 3;
    if (V[i][0] !== A[j][0] || Math.abs(V[i].length - A[j].length) > 3) return -2;
    let c = 0; for (const g of bv[i]) if (ba[j].has(g)) c++;
    return 2 * c / ((bv[i].size + ba[j].size) || 1) >= 0.6 ? 1.5 : -2;
  };
  const lo = i => Math.round(i * m / Math.max(n, 1)) - B, NEG = -1e9;
  const tb = new Uint8Array((n + 1) * W);
  let prev = new Float32Array(W).fill(NEG), cur = new Float32Array(W);
  for (let i = 0; i <= n; i++) {
    const L = lo(i), PL = i ? lo(i - 1) : 0;
    for (let k = 0; k < W; k++) {
      const j = L + k; let best = NEG, d = 0;
      if (j < 0 || j > m) { cur[k] = NEG; continue; }
      if (i === 0) { best = -j; d = 2; }
      else {
        const kd = j - 1 - PL, ku = j - PL;
        if (j > 0 && kd >= 0 && kd < W && prev[kd] > NEG / 2) { const v = prev[kd] + score(i - 1, j - 1); if (v > best) { best = v; d = 0; } }
        if (ku >= 0 && ku < W && prev[ku] > NEG / 2) { const v = prev[ku] - 1; if (v > best) { best = v; d = 1; } }
        if (k > 0 && cur[k - 1] > NEG / 2) { const v = cur[k - 1] - 1; if (v > best) { best = v; d = 2; } }
      }
      cur[k] = best; tb[i * W + k] = d;
    }
    [prev, cur] = [cur, prev];
  }
  const map = new Array(n).fill(-1);
  let i = n, j = m;
  while (i > 0 || j > 0) {
    const d = tb[i * W + (j - lo(i))];
    if (i > 0 && j > 0 && d === 0) { if (score(i - 1, j - 1) > 0) map[i - 1] = j - 1; i--; j--; }
    else if (i > 0 && (d === 1 || j === 0)) i--;
    else j--;
  }
  return map; // map[videoWord] = audioWord index, or -1
}

function align(segs, words) {
  const V = [], own = [], c0 = [];
  segs.forEach((s, si) => { c0[si] = V.length; s.text.split(' ').forEach(w => { V.push(w); own.push(si); }); });
  const map = dpAlign(V, words.map(x => x.w));
  const n = segs.length, ai = new Array(n).fill(NaN), tot = new Array(n).fill(0), hit = new Array(n).fill(0);
  V.forEach((_, vi) => {
    const si = own[vi]; tot[si]++;
    if (map[vi] >= 0) { hit[si]++; if (isNaN(ai[si])) ai[si] = Math.max(0, map[vi] - (vi - c0[si])); }
  });
  for (let i = 0; i < n; i++) if (isNaN(ai[i])) { // no matched word: interpolate between neighbours
    let p = i - 1; while (p >= 0 && isNaN(ai[p])) p--;
    let q = i + 1; while (q < n && isNaN(ai[q])) q++;
    const [cp, ap] = p >= 0 ? [c0[p], ai[p]] : [0, 0];
    const [cq, aq] = q < n ? [c0[q], ai[q]] : [V.length, words.length];
    ai[i] = ap + (c0[i] - cp) / Math.max(cq - cp, 1) * (aq - ap);
  }
  let pv = 0;
  segs.forEach((s, i) => {
    ai[i] = Math.min(Math.max(ai[i], pv), words.length - 1); pv = ai[i];
    s.ai = Math.round(ai[i]); s.as = words[s.ai].s; s.conf = tot[i] ? hit[i] / tot[i] : 0;
  });
  segs[0].as = 0;
  for (let i = 1; i < n; i++) segs[i].as = Math.max(segs[i].as, segs[i - 1].as + 0.3);
  if (atotal > 0) for (let i = n - 1; i > 0; i--) segs[i].as = Math.min(segs[i].as, (segs[i + 1] ? segs[i + 1].as : atotal) - 0.3);
  return hit.reduce((a, b) => a + b, 0) / Math.max(V.length, 1);
}

// FREE timing: find pauses in the audio with ffmpeg, then snap the audio-SRT line breaks to those pauses
async function findPauses(files, db) {
  const f = await getFF();
  await buildAudio(files, ['-ac', '1', '-ar', '16000', '-b:a', '32k'], 'mp3', 'w');
  const dur = await probe(f, 'w.mp3');
  await run(f, ['-i', 'w.mp3', '-af', `silencedetect=noise=${db}dB:d=0.3`, '-f', 'null', '-']);
  const gaps = []; let st = null;
  for (const m of tail) {
    const a = m.match(/silence_start:\s*(-?[\d.]+)/), b = m.match(/silence_end:\s*([\d.]+)/);
    if (a) st = Math.max(0, +a[1]);
    if (b && st !== null) { gaps.push({ s: st, e: +b[1] }); st = null; }
  }
  return { gaps, dur };
}

function retimeByPauses(cues, gaps, total) {
  const n = cues.length; if (n < 2) return null;
  let t0 = 0, tEnd = total; gaps = gaps.filter(g => g.e > g.s);
  if (gaps.length && gaps[0].s < 0.1) { t0 = gaps[0].e; gaps = gaps.slice(1); }
  if (gaps.length && gaps[gaps.length - 1].e >= total - 0.1) { tEnd = gaps[gaps.length - 1].s; gaps = gaps.slice(0, -1); }
  const G = gaps.length, K = n - 1; if (G < K) return null;
  let lost = 0;
  const sb = gaps.map(g => { const x = g.s - t0 - lost; lost += g.e - g.s; return x; }); // speech time before each pause
  const speech = tEnd - t0 - lost;
  const wc = cues.map(c => c.text.split(' ').length), rate = speech / wc.reduce((a, b) => a + b, 0);
  // a line's speech time should be close to (its word count x average time per word); pauses between lines are preferred long
  const cc = (k, x) => Math.abs(Math.log(Math.max(x, 0.05) / (wc[k] * rate)));
  const gc = g => -0.15 * Math.min(gaps[g].e - gaps[g].s, 1.5) / 1.5;
  const dp = Array.from({ length: K }, () => new Float64Array(G).fill(Infinity)), bk = Array.from({ length: K }, () => new Int32Array(G).fill(-1));
  for (let g = 0; g < G; g++) dp[0][g] = cc(0, sb[g]) + gc(g);
  for (let k = 1; k < K; k++)
    for (let g = k; g < G; g++) {
      let best = Infinity, bi = -1;
      for (let h = k - 1; h < g; h++) { const v = dp[k - 1][h] + cc(k, sb[g] - sb[h]); if (v < best) { best = v; bi = h; } }
      if (bi >= 0) { dp[k][g] = best + gc(g); bk[k][g] = bi; }
    }
  let g = -1, bv = Infinity;
  for (let i = 0; i < G; i++) { const v = dp[K - 1][i] + cc(n - 1, speech - sb[i]); if (v < bv) { bv = v; g = i; } }
  if (g < 0) return null;
  const sel = new Array(K); sel[K - 1] = g;
  for (let k = K - 1; k > 0; k--) sel[k - 1] = bk[k][sel[k]];
  const starts = [t0, ...sel.map(i => gaps[i].e)], ends = [...sel.map(i => gaps[i].s), tEnd];
  const words = [];
  cues.forEach((c, k) => { const ws = c.text.split(' '), d = (ends[k] - starts[k]) / ws.length; ws.forEach((w, i) => words.push({ w, s: starts[k] + i * d })); });
  return { words, starts };
}

// Re-time the audio-SRT words using real word times from Whisper (SRT times are often only approximate)
function retime(sw, ww) {
  const m = dpAlign(sw.map(x => x.w), ww.map(x => x.w)), anc = [];
  m.forEach((j, k) => { if (j >= 0) anc.push([k, ww[j].s]); });
  const frac = anc.length / Math.max(sw.length, 1);
  if (anc.length < Math.max(5, 0.1 * sw.length)) return { words: sw, frac };
  let p = 0;
  const words = sw.map((x, k) => {
    while (p + 1 < anc.length && anc[p + 1][0] <= k) p++;
    const a = anc[p], b = anc[p + 1];
    const t = k <= anc[0][0] ? anc[0][1] - (anc[0][0] - k) * 0.35
      : !b ? a[1] + (k - a[0]) * 0.35
      : a[1] + (k - a[0]) / (b[0] - a[0]) * (b[1] - a[1]);
    return { w: x.w, s: Math.max(0, t) };
  });
  return { words, frac };
}

const aEnd = i => segs[i + 1] ? segs[i + 1].as : atotal;
const lim = () => [+$('minS').value, +$('maxS').value];

function draw() {
  const [mn, mx] = lim();
  const num = (s, k, i) => `<td><input type=number step=0.05 value="${s[k].toFixed(2)}" data-k="${k}" data-i="${i}">`;
  $('tbl').innerHTML = '<tr><th>#<th>Caption<th>Audio says (auto)<th>Video start<th>Video end<th>Audio start<th>Speed<th>Match</tr>' +
    segs.map((s, i) => {
      const f = (s.ve - s.vs) / Math.max(aEnd(i) - s.as, 0.1);
      return `<tr><td>${i + 1}<td>${esc(s.text.slice(0, 40))}<td>${esc(awords.slice(s.ai, s.ai + 6).map(x => x.w).join(' '))}${num(s, 'vs', i)}${num(s, 've', i)}${num(s, 'as', i)}` +
        `<td>${f.toFixed(2)}x${f < mn || f > mx ? ' ⚠' : ''}<td>${Math.round(s.conf * 100)}%${s.conf < .5 ? ' ⚠' : ''}</tr>`;
    }).join('');
}
$('tbl').onchange = e => { segs[e.target.dataset.i][e.target.dataset.k] = +e.target.value; draw(); };

$('analyze').onclick = async () => {
  ready = false;
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
      log('Using audio SRT text…');
      atotal = 0; for (const f of af) atotal += (await meta(f)).dur;
      const cues = parseSrt(await asrt.text()); words = wordsFromSrt(cues);
      if (cues.length) atotal = Math.max(atotal, cues[cues.length - 1].e);
      const tm = $('tm').value;
      if (tm === 'whisper') {
        if (!$('key').value.trim()) throw new Error('Whisper timing needs your OpenAI key.');
        log('Measuring real audio timing with Whisper (SRT times may be approximate)…');
        const tr = await transcribe(af), r = retime(words, tr.words);
        words = r.words; atotal = tr.total;
        log(`Whisper timing matched ${Math.round(r.frac * 100)}% of the SRT words` + (r.frac < 0.3 ? ' (too low, keeping the SRT times).' : '.'));
      } else if (tm === 'free') {
        log('Detecting pauses in the audio to measure the real timing (free)…');
        const { gaps, dur } = await findPauses(af, +$('db').value || -35);
        const r = dur > 0 ? retimeByPauses(cues, gaps, dur) : null;
        if (r) { words = r.words; atotal = dur; log(`Found ${gaps.length} pauses and placed ${cues.length} lines on them.`); }
        else log(`Only ${gaps.length} pauses found for ${cues.length} lines, so the SRT times are kept. Try a higher dB value (for example -30).`);
      } else log('Using the SRT times exactly as given.');
    } else {
      log('Transcribing audio…');
      const tr = await transcribe(af); atotal = tr.total; words = tr.words;
    }
    if (!segs.length || !words.length) throw new Error('No captions or no spoken words found.');
    awords = words;
    const frac = align(segs, words); draw();
    log(`Matched ${Math.round(frac * 100)}% of caption words to the audio.` + (frac < 0.4 ? ' LOW: the two texts differ a lot, so check every ⚠ row.' : ''));
    if (frac < 0.15) { log('STOP: the audio text and the video captions are almost completely different (probably the wrong audio SRT or audio file). Upload the matching audio SRT and click Analyze again.'); return; }
    ready = true; log('Review the table (fix anything marked ⚠), then Render.');
  } catch (e) { fail(e); }
};

// ---------- 4. Render: per-scene speed change, hard cuts ----------
$('render').onclick = async () => {
  try {
    const f = await getFF(), { w, h } = vinfo, [mn, mx] = lim(), mode = $('pad').value, FPS = 30;
    const w2 = w - (w % 2), h2 = h - (h % 2);
    const inp = await mountVideo($('video').files[0]);
    if (mode === 'image' && $('img').files[0]) await f.writeFile('img', await fetchFile($('img').files[0]));
    log('Preparing audio…');
    await buildAudio([...$('audio').files], ['-ar', '44100', '-ac', '2', '-b:a', '160k'], 'm4a', 'h');
    const adur = await probe(f, 'h.m4a'); if (adur > 0) atotal = adur;
    const list = [];
    const fill = mode === 'image' && !$('img').files[0] ? 'freeze' : mode;
    for (let i = 0; i < segs.length; i++) {
      const s = segs[i], vd = s.ve - s.vs;
      // whole frames, so the scenes add up exactly to the audio length
      const N = Math.max(1, Math.round(FPS * aEnd(i)) - Math.round(FPS * s.as)), td = N / FPS;
      const sp = Math.min(mx, Math.max(mn, vd / td)), got = vd / sp, gap = td - got, pad = gap > 0.04;
      const nMain = pad && fill === 'image' ? Math.max(1, Math.round(got * FPS)) : N;
      let vf = `setpts=PTS/${sp},fps=${FPS},scale=${w2}:${h2},setsar=1`;
      if (pad && fill === 'freeze') vf += `,tpad=stop_mode=clone:stop_duration=${gap + 0.2}`;
      if (pad && fill === 'black') vf += `,tpad=stop_mode=add:stop_duration=${gap + 0.2}:color=black`;
      await run(f, ['-ss', String(s.vs), '-t', String(vd), '-i', inp, '-vf', vf, '-an', '-frames:v', String(nMain),
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '26', '-pix_fmt', 'yuv420p', '-r', String(FPS), `s${i}.mp4`]);
      list.push(`s${i}.mp4`);
      if (pad && fill === 'image' && N - nMain > 0) {
        await run(f, ['-loop', '1', '-framerate', String(FPS), '-i', 'img', '-frames:v', String(N - nMain), '-vf',
          `scale=${w2}:${h2}:force_original_aspect_ratio=decrease,pad=${w2}:${h2}:(ow-iw)/2:(oh-ih)/2,fps=${FPS},setsar=1`,
          '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '26', '-pix_fmt', 'yuv420p', `p${i}.mp4`]);
        list.push(`p${i}.mp4`);
      }
      log(`Scene ${i + 1}/${segs.length}: ${sp.toFixed(2)}x${pad ? ', +' + gap.toFixed(1) + 's fill' : ''}`);
    }
    log('Joining scenes…');
    await f.writeFile('v.txt', list.map(n => `file '${n}'`).join('\n'));
    await run(f, ['-f', 'concat', '-safe', '0', '-i', 'v.txt', '-c', 'copy', 'video.mp4']);
    const vdur = await probe(f, 'video.mp4');
    log(`Video ${vdur.toFixed(1)}s, audio ${atotal.toFixed(1)}s`);
    let finalVideo = 'video.mp4';
    const gap = atotal - vdur;
    if (gap > 0.1) { // video ended early: hold the last frame (a tiny clip, no re-encode of the whole video)
      await run(f, ['-sseof', '-0.05', '-i', 'video.mp4', '-vframes', '1', 'last.png']);
      await run(f, ['-loop', '1', '-framerate', String(FPS), '-i', 'last.png', '-frames:v', String(Math.ceil(gap * FPS)),
        '-vf', `scale=${w2}:${h2},fps=${FPS},setsar=1`, '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '26', '-pix_fmt', 'yuv420p', 'tail.mp4']);
      await f.writeFile('v2.txt', "file 'video.mp4'\nfile 'tail.mp4'");
      await run(f, ['-f', 'concat', '-safe', '0', '-i', 'v2.txt', '-c', 'copy', 'video2.mp4']);
      finalVideo = 'video2.mp4';
    }
    log('Adding audio…');
    await run(f, ['-i', finalVideo, '-i', 'h.m4a', '-map', '0:v', '-map', '1:a', '-c', 'copy', '-shortest', 'out.mp4']);
    const url = URL.createObjectURL(new Blob([await f.readFile('out.mp4')], { type: 'video/mp4' }));
    for (const n of list) { try { await f.deleteFile(n); } catch {} }
    $('out').src = url; $('dl').href = url; log('Done.');
  } catch (e) { fail(e); }
};

// one job at a time: ffmpeg.wasm can't run two jobs on the same instance
const once = fn => async () => {
  if (busy) return; busy = true; $('analyze').disabled = $('render').disabled = true;
  try { await fn(); } finally { busy = false; $('analyze').disabled = false; $('render').disabled = !ready; }
};
$('analyze').onclick = once($('analyze').onclick);
$('render').onclick = once($('render').onclick);
