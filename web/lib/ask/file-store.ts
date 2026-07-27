/**
 * Device-side storage for Ask Anything attachment payloads (slice C).
 * Transcripts stay in localStorage (`aa-chats-v1`), but raw file bytes (PDFs,
 * images) do not fit its ~5 MB quota, so their data URLs live here in
 * IndexedDB and the persisted message carries only an `aa-file:<id>`
 * reference. Loading a chat rehydrates references back into real data URLs;
 * deleting a chat deletes its files. ADR-022 holds: nothing leaves the device.
 *
 * Every function is best-effort: where IndexedDB is unavailable (private
 * mode, exotic browsers), callers keep working for the live session and the
 * attachment simply does not survive a reload.
 */

import { FILE_REF_PREFIX } from "@/lib/files/attachments";

export interface StoredFile {
  id: string;
  chatId: string;
  name: string;
  mediaType: string;
  dataUrl: string;
}

const DB_NAME = "aa-files-v1";
const STORE = "files";

export function fileRef(id: string): string {
  return `${FILE_REF_PREFIX}${id}`;
}

export function isFileRef(url: unknown): url is string {
  return typeof url === "string" && url.startsWith(FILE_REF_PREFIX);
}

export function idFromRef(ref: string): string {
  return ref.slice(FILE_REF_PREFIX.length);
}

function openDb(): Promise<IDBDatabase | null> {
  return new Promise((resolve) => {
    try {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE)) {
          const store = db.createObjectStore(STORE, { keyPath: "id" });
          store.createIndex("chatId", "chatId", { unique: false });
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
      req.onblocked = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

/** Runs one transaction; resolves null/false instead of rejecting. */
async function withStore<T>(
  mode: IDBTransactionMode,
  work: (store: IDBObjectStore) => IDBRequest<T> | IDBRequest[],
): Promise<T | null> {
  const db = await openDb();
  if (!db) return null;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE, mode);
      const result = work(tx.objectStore(STORE));
      tx.oncomplete = () => {
        db.close();
        resolve(Array.isArray(result) ? (undefined as T) : result.result);
      };
      tx.onerror = () => {
        db.close();
        resolve(null);
      };
      tx.onabort = () => {
        db.close();
        resolve(null);
      };
    } catch {
      db.close();
      resolve(null);
    }
  });
}

/** Stores one attachment payload. Returns false when storage is unavailable
 * (quota, private mode), in which case the attachment is session-only. */
export async function putFile(file: StoredFile): Promise<boolean> {
  const result = await withStore("readwrite", (store) => store.put(file));
  return result !== null;
}

export async function getFile(id: string): Promise<StoredFile | null> {
  const result = await withStore<StoredFile | undefined>("readonly", (store) =>
    store.get(id) as IDBRequest<StoredFile | undefined>,
  );
  return result ?? null;
}

/** Removes every stored payload belonging to a chat (called on chat delete). */
export async function deleteFilesForChat(chatId: string): Promise<void> {
  await withStore("readwrite", (store) => {
    const index = store.index("chatId");
    const req = index.openCursor(IDBKeyRange.only(chatId));
    req.onsuccess = () => {
      const cursor = req.result;
      if (cursor) {
        cursor.delete();
        cursor.continue();
      }
    };
    return [req];
  });
}
