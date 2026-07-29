import { BUILD_DATE } from "@/lib/config/build-info";
import { version } from "@/package.json";

export function SiteFooter() {
  return (
    <footer className="mt-16 border-t border-medium-tan bg-light-tan">
      <div className="mx-auto grid max-w-6xl gap-8 px-4 py-10 text-sm sm:grid-cols-3">
        <section aria-labelledby="footer-use">
          <h2 id="footer-use" className="mb-2 text-base">
            Responsible use
          </h2>
          <ul className="list-disc space-y-1 pl-5 text-ink">
            <li>ChatISA is for educational purposes only.</li>
            <li>Get instructor approval before using it for classwork.</li>
            <li>Evaluate every answer critically. Models make mistakes.</li>
          </ul>
        </section>
        <section aria-labelledby="footer-people">
          <h2 id="footer-people" className="mb-2 text-base">
            Maintained by
          </h2>
          <ul className="space-y-1">
            <li>
              <a
                className="font-bold text-accent-red underline underline-offset-2"
                href="https://miamioh.edu/fsb/directory/?up=/directory/megahefm"
              >
                Fadel Megahed
              </a>
            </li>
          </ul>
          <p className="mt-2 text-dark-tan">
            Farmer School of Business, Miami University
          </p>
        </section>
        <section aria-labelledby="footer-support">
          <h2 id="footer-support" className="mb-2 text-base">
            Supported by
          </h2>
          <ul className="space-y-1 text-ink">
            <li>U.S. Bank (API costs)</li>
            <li>Raymond E. Glos Professorship</li>
            <li>Farmer School of Business IT Office</li>
          </ul>
          <p className="mt-3">
            <a
              className="font-bold text-accent-red underline underline-offset-2"
              href="https://arxiv.org/abs/2407.15010"
            >
              Read the ChatISA research paper
            </a>
          </p>
        </section>
      </div>
      <div className="border-t border-medium-tan">
        <p className="mx-auto max-w-6xl px-4 py-3 text-xs text-dark-tan">
          ChatISA {CHATISA_VERSION} · Updated {CHATISA_UPDATED}
        </p>
      </div>
    </footer>
  );
}

/** Derived from package.json so a release version bump can never miss the
 * footer again (v6.1.0 and v6.1.1 shipped still displaying v6.0.0). The date
 * beside it is stamped automatically when the deploy bundle is made; see
 * lib/config/build-info. */
const CHATISA_VERSION = `v${version}`;
const CHATISA_UPDATED = BUILD_DATE;
