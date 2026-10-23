// A small animated-GIF writer for scripts/demo-video.ts, so the README's preview needs no image
// library. It reads the PNG frames that Playwright's bundled ffmpeg extracts from the recording
// (8-bit RGB or RGBA, not interlaced), picks one 255-colour palette for the whole animation, and
// writes each frame as only the rectangle that changed, with unchanged pixels transparent.
import { readFileSync } from 'node:fs'
import { inflateSync } from 'node:zlib'

export type Rgb = { width: number; height: number; data: Uint8Array } // 3 bytes per pixel

/** Decode a non-interlaced 8-bit RGB or RGBA PNG into RGB. */
export function readPng(path: string): Rgb {
  const buf = readFileSync(path)
  let width = 0
  let height = 0
  let channels = 0
  const idat: Buffer[] = []
  for (let at = 8; at < buf.length; ) {
    const length = buf.readUInt32BE(at)
    const type = buf.toString('latin1', at + 4, at + 8)
    const body = buf.subarray(at + 8, at + 8 + length)
    if (type === 'IHDR') {
      width = body.readUInt32BE(0)
      height = body.readUInt32BE(4)
      const [depth, colour, , , interlace] = body.subarray(8, 13)
      if (depth !== 8 || (colour !== 2 && colour !== 6) || interlace !== 0) throw new Error(`${path}: unsupported PNG`)
      channels = colour === 2 ? 3 : 4
    } else if (type === 'IDAT') idat.push(body)
    at += 12 + length
  }
  const raw = inflateSync(Buffer.concat(idat))
  const stride = width * channels
  const px = new Uint8Array(stride * height)
  for (let y = 0; y < height; y += 1) {
    const filter = raw[y * (stride + 1)]
    const src = y * (stride + 1) + 1
    const row = y * stride
    for (let x = 0; x < stride; x += 1) {
      const a = x >= channels ? px[row + x - channels] : 0 // left
      const b = y > 0 ? px[row - stride + x] : 0 // up
      const c = x >= channels && y > 0 ? px[row - stride + x - channels] : 0 // up-left
      let v = raw[src + x]
      if (filter === 1) v += a
      else if (filter === 2) v += b
      else if (filter === 3) v += (a + b) >> 1
      else if (filter === 4) {
        const p = a + b - c
        const pa = Math.abs(p - a)
        const pb = Math.abs(p - b)
        const pc = Math.abs(p - c)
        v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c
      }
      px[row + x] = v & 0xff
    }
  }
  if (channels === 3) return { width, height, data: px }
  const rgb = new Uint8Array(width * height * 3)
  for (let i = 0, j = 0; i < px.length; i += 4, j += 3) rgb.set(px.subarray(i, i + 3), j)
  return { width, height, data: rgb }
}

const TRANSPARENT = 255 // palette index left for "unchanged since the last frame"
const NOISE = 36 // summed |ΔR| + |ΔG| + |ΔB| below which a pixel counts as unchanged
const key15 = (d: Uint8Array, i: number) => ((d[i] >> 3) << 10) | ((d[i + 1] >> 3) << 5) | (d[i + 2] >> 3)

/**
 * One palette for every frame: the 255 most common colours (at 5 bits a channel) across the samples,
 * each the mean of the pixels it stands for, and a lookup from every 15-bit colour to its nearest entry.
 */
export function buildPalette(samples: Rgb[]): { palette: Uint8Array; lookup: Uint8Array } {
  const count = new Float64Array(32768)
  const sum = new Float64Array(32768 * 3)
  for (const { data } of samples) {
    for (let i = 0; i < data.length; i += 3) {
      const k = key15(data, i)
      count[k] += 1
      sum[k * 3] += data[i]
      sum[k * 3 + 1] += data[i + 1]
      sum[k * 3 + 2] += data[i + 2]
    }
  }
  const keys = [...count.keys()].filter((k) => count[k] > 0).sort((p, q) => count[q] - count[p]).slice(0, 255)
  const palette = new Uint8Array(256 * 3)
  keys.forEach((k, j) => {
    for (let ch = 0; ch < 3; ch += 1) palette[j * 3 + ch] = Math.round(sum[k * 3 + ch] / count[k])
  })
  const lookup = new Uint8Array(32768)
  for (let k = 0; k < 32768; k += 1) {
    const r = ((k >> 10) << 3) + 4
    const g = (((k >> 5) & 31) << 3) + 4
    const b = ((k & 31) << 3) + 4
    let best = 0
    let bestD = Infinity
    for (let j = 0; j < keys.length; j += 1) {
      const dr = r - palette[j * 3]
      const dg = g - palette[j * 3 + 1]
      const db = b - palette[j * 3 + 2]
      const d = 2 * dr * dr + 4 * dg * dg + 3 * db * db
      if (d < bestD) {
        bestD = d
        best = j
      }
    }
    lookup[k] = best
  }
  return { palette, lookup }
}

/** GIF's variable-length LZW, with 8-bit symbols, packed into 255-byte sub-blocks. */
function lzw(indices: Uint8Array): Buffer {
  const clear = 256
  const eoi = 257
  const bytes: number[] = []
  let acc = 0
  let bits = 0
  let size = 9
  const emit = (code: number) => {
    acc |= code << bits
    bits += size
    while (bits >= 8) {
      bytes.push(acc & 0xff)
      acc >>>= 8
      bits -= 8
    }
  }
  let table = new Map<number, number>()
  let next = eoi + 1
  emit(clear)
  let prefix = indices[0]
  for (let i = 1; i < indices.length; i += 1) {
    const k = indices[i]
    const code = table.get((prefix << 8) | k)
    if (code !== undefined) {
      prefix = code
      continue
    }
    emit(prefix)
    if (next === 4096) {
      emit(clear)
      table = new Map()
      next = eoi + 1
      size = 9
    } else {
      if (next >= 1 << size) size += 1
      table.set((prefix << 8) | k, next)
      next += 1
    }
    prefix = k
  }
  emit(prefix)
  emit(eoi)
  if (bits > 0) bytes.push(acc & 0xff)
  const out: number[] = [8]
  for (let i = 0; i < bytes.length; i += 255) {
    const block = bytes.slice(i, i + 255)
    out.push(block.length, ...block)
  }
  out.push(0)
  return Buffer.from(out)
}

const u16 = (v: number) => [v & 0xff, v >> 8]

/** Writes frames one at a time; each is shown until the next one's time. Returns the GIF's bytes. */
export class GifWriter {
  private parts: Buffer[] = []
  private shown: Uint8Array | null = null // palette index of each pixel on screen
  private pending: { at: number; bytes: Buffer[] } | null = null
  private width: number
  private height: number
  private lookup: Uint8Array
  private palette: Uint8Array

  constructor(width: number, height: number, palette: Uint8Array, lookup: Uint8Array) {
    this.width = width
    this.height = height
    this.lookup = lookup
    this.palette = palette
    this.parts.push(
      Buffer.from('GIF89a', 'latin1'),
      Buffer.from([...u16(width), ...u16(height), 0xf7, 0, 0]),
      Buffer.from(palette),
      Buffer.from([0x21, 0xff, 0x0b, ...Buffer.from('NETSCAPE2.0', 'latin1'), 3, 1, 0, 0, 0]), // loop forever
    )
  }

  /**
   * Add the frame shown from `at` seconds. A pixel counts as changed only when its colour moved more
   * than NOISE from what is on screen, so the video's compression flicker isn't redrawn; a frame
   * with no changed pixel only extends the one before.
   */
  add(frame: Rgb, at: number) {
    const { width, height, lookup, palette } = this
    const first = this.shown === null
    const shown = (this.shown ??= new Uint8Array(width * height))
    const changed = new Uint8Array(width * height)
    let [x0, y0, x1, y1] = [width, height, -1, -1]
    for (let y = 0, p = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1, p += 1) {
        const i = p * 3
        const j = lookup[key15(frame.data, i)]
        const s = shown[p] * 3
        const moved =
          Math.abs(frame.data[i] - palette[s]) + Math.abs(frame.data[i + 1] - palette[s + 1]) + Math.abs(frame.data[i + 2] - palette[s + 2])
        if (first || (j !== shown[p] && moved > NOISE)) {
          changed[p] = 1
          shown[p] = j
          if (x < x0) x0 = x
          if (x > x1) x1 = x
          if (y < y0) y0 = y
          if (y > y1) y1 = y
        }
      }
    }
    if (x1 < 0) return // nothing changed
    const w = x1 - x0 + 1
    const h = y1 - y0 + 1
    const sub = new Uint8Array(w * h)
    for (let y = 0; y < h; y += 1) {
      for (let x = 0; x < w; x += 1) {
        const p = (y0 + y) * width + x0 + x
        sub[y * w + x] = changed[p] ? shown[p] : TRANSPARENT
      }
    }
    this.flush(at)
    this.pending = {
      at,
      bytes: [Buffer.from([0x2c, ...u16(x0), ...u16(y0), ...u16(w), ...u16(h), 0]), lzw(sub)],
    }
  }

  /** The finished GIF, whose last frame shows until `end` seconds. */
  finish(end: number): Buffer {
    this.flush(end)
    this.parts.push(Buffer.from([0x3b]))
    return Buffer.concat(this.parts)
  }

  private flush(until: number) {
    if (this.pending === null) return
    // Delays are in hundredths; rounding the start times keeps the total in step with the video.
    const delay = Math.max(2, Math.round(until * 100) - Math.round(this.pending.at * 100))
    const transparent = this.parts.length > 4 ? 1 : 0 // the first frame is drawn whole
    this.parts.push(Buffer.from([0x21, 0xf9, 4, (1 << 2) | transparent, ...u16(delay), TRANSPARENT, 0]), ...this.pending.bytes)
    this.pending = null
  }
}
