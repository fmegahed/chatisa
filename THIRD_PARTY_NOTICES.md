# Third-party notices

ChatISA's own source code is MIT-licensed (see `LICENSE`). This file lists
third-party software that ChatISA serves to students' browsers as separate
programs, under their own licences.

## MAIDR (chart explorer)

- **What it is:** MAIDR, "Multimodal Access and Interactive Data
  Representation" (https://maidr.ai), from the xability lab. It makes a plot
  explorable from the keyboard, with each value announced, played as a tone, or
  shown in braille. ChatISA uses it for plots in the Coding Studio, in runnable
  code in chat replies, and in Ask Anything (issue #40).
- **Version:** 4.16.0 (npm package `maidr`, published 2026-10-19).
- **Licence:** GPL-3.0-or-later. The full text is served beside the script at
  `/runtimes/maidr/4.16.0/LICENSE`.
- **How ChatISA uses it:** as a separate, unmodified file. `maidr.js`, its
  locale packs and its CSS are copied byte for byte from the npm package by
  `web/scripts/setup-runtimes.mjs`, which checks the package's published
  integrity and the script's own hash against `web/lib/run/maidr-pin.json`.
  They are served from ChatISA's origin and loaded at runtime by a script tag
  when a navigable plot first appears. ChatISA's code never imports or bundles
  MAIDR (a unit test checks this); it only writes the plain chart data that
  maidr.js reads.
- **Corresponding source:**
  https://github.com/xability/maidr/tree/21cb21908b774d95222544d580387bf5aa2dd2da
  (the commit the 4.16.0 package was built from).
- **Package:** https://registry.npmjs.org/maidr/-/maidr-4.16.0.tgz,
  integrity
  `sha512-aIGy56JWpiWqY35dqAzNmkhfTZPAQsVaaZzv4p3H4wb/tyCaIPIcQjgGeox2nX3qE8gwZXd0Kxy9pla2XOVrZQ==`.
- **Network use:** none by default; every MAIDR file comes from ChatISA's
  origin. Two optional features reach elsewhere only when a student turns them
  on: support for a DotPad tactile display loads its SDK from jsDelivr, and
  MAIDR's AI chat (off by default) sends the chart to the provider whose API
  key the student enters in MAIDR's own settings.
