# Caption Sync
Matches a captioned video to separate audio: OCR reads the on-screen captions, Whisper transcribes the audio,
each caption scene is aligned to its spoken line, and every scene is sped up / slowed down (with hard cuts) to fit.

## Deploy
1. Push this folder to a GitHub repo.
2. On vercel.com: Add New > Project > import the repo. Framework is auto-detected (Vite). No env vars needed.

## Notes
- Everything runs in the browser (ffmpeg.wasm + Tesseract.js). Your OpenAI key is typed into the page and sent only to OpenAI.
- Best for videos up to a few minutes / a few hundred MB. For longer videos, move the render step to a server worker.
- Run locally: `npm install && npm run dev`
