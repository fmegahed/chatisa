/**
 * Browser-side photo preparation. The student's original never leaves the
 * device; a 512 px JPEG re-encode is what gets pushed, which keeps the repo
 * small and strips EXIF (location data included).
 */

export function fitWithin(width: number, height: number, maxSide: number): { width: number; height: number } {
  if (width <= maxSide && height <= maxSide) return { width, height };
  const scale = maxSide / Math.max(width, height);
  return { width: Math.round(width * scale), height: Math.round(height * scale) };
}

export function dataUrlToBase64(dataUrl: string): string {
  const i = dataUrl.indexOf(",");
  return i === -1 ? dataUrl : dataUrl.slice(i + 1);
}

function base64Bytes(b64: string): number {
  return Math.floor((b64.replace(/=+$/, "").length * 3) / 4);
}

export async function resizePhoto(
  file: File,
  opts: { maxSide?: number; maxBytes?: number } = {},
): Promise<{ base64: string; bytes: number; width: number; height: number }> {
  const maxSide = opts.maxSide ?? 512;
  const maxBytes = opts.maxBytes ?? 150_000;
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    throw new Error("That image could not be read.");
  }
  const { width, height } = fitWithin(bitmap.width, bitmap.height, maxSide);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("That image could not be read.");
  ctx.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  // Step quality down until the JPEG fits; 0.5 is the floor.
  for (const quality of [0.85, 0.75, 0.65, 0.5]) {
    const base64 = dataUrlToBase64(canvas.toDataURL("image/jpeg", quality));
    const bytes = base64Bytes(base64);
    if (bytes <= maxBytes || quality === 0.5) return { base64, bytes, width, height };
  }
  throw new Error("That image could not be read.");
}
