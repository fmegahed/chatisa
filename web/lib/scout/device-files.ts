/**
 * Device-side storage for Job Scout's larger payloads: the student's resume
 * PDF and generated project scaffolds. Modeled on lib/ask/file-store.ts:
 * metadata lives in localStorage, bytes-as-data-URLs live here in IndexedDB
 * because they do not fit localStorage's ~5 MB quota. Nothing here ever
 * leaves the device (local-first decision, 2026-07-28; resume persistence
 * approved by user 2026-07-29).
 *
 * Every function is best-effort: where IndexedDB is unavailable (private
 * mode, exotic browsers), callers keep working for the live session and the
 * stored copy simply does not survive a reload.
 */

const DB_NAME = "js-files-v1";
const STORE = "files";
const RESUME_KEY = "resume";

export interface DeviceResume {
  name: string;
  dataUrl: string;
  addedAt: string;
}

interface StoredRecord {
  id: string;
  json: string;
}

function openDb(): Promise<IDBDatabase | null> {
  return new Promise((resolve) => {
    try {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = () => {
        req.result.createObjectStore(STORE, { keyPath: "id" });
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

export async function putItem(id: string, value: unknown): Promise<boolean> {
  const db = await openDb();
  if (!db) return false;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).put({ id, json: JSON.stringify(value) });
      tx.oncomplete = () => {
        db.close();
        resolve(true);
      };
      tx.onerror = () => {
        db.close();
        resolve(false);
      };
    } catch {
      db.close();
      resolve(false);
    }
  });
}

export async function getItem<T>(id: string): Promise<T | null> {
  const db = await openDb();
  if (!db) return null;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE, "readonly");
      const req = tx.objectStore(STORE).get(id);
      req.onsuccess = () => {
        db.close();
        const record = req.result as StoredRecord | undefined;
        if (!record) return resolve(null);
        try {
          resolve(JSON.parse(record.json) as T);
        } catch {
          resolve(null);
        }
      };
      req.onerror = () => {
        db.close();
        resolve(null);
      };
    } catch {
      db.close();
      resolve(null);
    }
  });
}

export async function removeItem(id: string): Promise<void> {
  const db = await openDb();
  if (!db) return;
  await new Promise<void>((resolve) => {
    try {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).delete(id);
      tx.oncomplete = () => {
        db.close();
        resolve();
      };
      tx.onerror = () => {
        db.close();
        resolve();
      };
    } catch {
      db.close();
      resolve();
    }
  });
}

// ------------------------------------------------------------------ resume

export async function putResume(name: string, dataUrl: string): Promise<boolean> {
  return putItem(RESUME_KEY, {
    name,
    dataUrl,
    addedAt: new Date().toISOString(),
  } satisfies DeviceResume);
}

export async function getResume(): Promise<DeviceResume | null> {
  return getItem<DeviceResume>(RESUME_KEY);
}

export async function deleteResume(): Promise<void> {
  return removeItem(RESUME_KEY);
}

/** Rehydrates the stored resume into a real File for a form upload. */
export async function resumeAsFile(): Promise<File | null> {
  const stored = await getResume();
  if (!stored) return null;
  try {
    const blob = await (await fetch(stored.dataUrl)).blob();
    return new File([blob], stored.name, { type: "application/pdf" });
  } catch {
    return null;
  }
}

// --------------------------------------------------------------- scaffolds

export async function putScaffold(projectId: string, scaffold: unknown): Promise<boolean> {
  return putItem(`scaffold:${projectId}`, scaffold);
}

export async function getScaffold<T>(projectId: string): Promise<T | null> {
  return getItem<T>(`scaffold:${projectId}`);
}

export async function deleteScaffold(projectId: string): Promise<void> {
  return removeItem(`scaffold:${projectId}`);
}
