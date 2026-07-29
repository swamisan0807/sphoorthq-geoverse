import { useEffect, useState } from "react";
import { fetchAuthedImageUrl } from "./client";

/** Loads an image that requires the Authorization header (can't just be an
 * <img src> string) as a blob object URL, revoking the previous one on
 * every change/unmount so we don't leak memory. */
export function useAuthedImage(path: string | null): { url: string | null; error: string | null } {
  const [url, setUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!path) {
      setUrl(null);
      return;
    }
    let cancelled = false;
    let objectUrl: string | null = null;
    setError(null);

    fetchAuthedImageUrl(path)
      .then((u) => {
        if (cancelled) {
          URL.revokeObjectURL(u);
          return;
        }
        objectUrl = u;
        setUrl(u);
      })
      .catch((e) => !cancelled && setError(String(e)));

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [path]);

  return { url, error };
}
