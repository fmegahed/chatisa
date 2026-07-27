/**
 * Client-side image preparation for Ask Anything attachments (slice C).
 * Providers charge vision by pixel area and reject oversized payloads, so
 * anything larger than the target is downscaled on a canvas before it becomes
 * a file part. The sizing decision is pure and unit-tested; the canvas work is
 * browser-only.
 */

/** Longest edge sent to a model. Frontier vision models see no benefit beyond
 * roughly this size, and tokens scale with area. */
export const IMAGE_MAX_DIM = 1600;

/** An image at most this large with in-range dimensions is sent as-is, which
 * preserves PNG transparency and GIF animation. */
export const IMAGE_KEEP_BYTES = 1_500_000;

/** JPEG quality for downscaled output. */
export const IMAGE_JPEG_QUALITY = 0.85;

/** Pure sizing decision: final dimensions and whether scaling is needed. */
export function targetDimensions(
  width: number,
  height: number,
  maxDim: number = IMAGE_MAX_DIM,
): { width: number; height: number; scaled: boolean } {
  const longest = Math.max(width, height);
  if (longest <= maxDim) return { width, height, scaled: false };
  const factor = maxDim / longest;
  return {
    width: Math.max(1, Math.round(width * factor)),
    height: Math.max(1, Math.round(height * factor)),
    scaled: true,
  };
}

/** Whether the original bytes can ride unchanged. */
export function keepOriginal(
  sizeBytes: number,
  width: number,
  height: number,
): boolean {
  return (
    sizeBytes <= IMAGE_KEEP_BYTES &&
    !targetDimensions(width, height).scaled
  );
}

function fileToDataUrl(file: File | Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("The image could not be read."));
    reader.readAsDataURL(file);
  });
}

/**
 * Browser-only: returns the data URL and media type to put on the file part.
 * Small in-range images pass through untouched; everything else is redrawn at
 * the target size and re-encoded as JPEG.
 */
export async function prepareImage(
  file: File,
): Promise<{ dataUrl: string; mediaType: string }> {
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    throw new Error(`"${file.name}" could not be decoded as an image.`);
  }
  try {
    if (keepOriginal(file.size, bitmap.width, bitmap.height)) {
      return { dataUrl: await fileToDataUrl(file), mediaType: file.type };
    }
    const { width, height } = targetDimensions(bitmap.width, bitmap.height);
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("This browser cannot resize images.");
    // White backdrop: JPEG has no alpha, and transparent PNGs would otherwise
    // composite onto black.
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);
    ctx.drawImage(bitmap, 0, 0, width, height);
    return {
      dataUrl: canvas.toDataURL("image/jpeg", IMAGE_JPEG_QUALITY),
      mediaType: "image/jpeg",
    };
  } finally {
    bitmap.close();
  }
}
