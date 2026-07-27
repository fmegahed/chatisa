/**
 * The ws-proxy that WebR routes R's libcurl through, so rvest/httr2/curl can
 * reach the internet (including sites without CORS headers). The value is
 * public, not a secret. Defaults to the public rOpenSci proxy; override at build
 * time with NEXT_PUBLIC_WS_PROXY. A runtime-swappable value ships with the
 * self-hosted proxy (see roadmap.md).
 */
export const WS_PROXY =
  process.env.NEXT_PUBLIC_WS_PROXY ?? "socks5h://test:yolo@ws.r-universe.dev:443";
