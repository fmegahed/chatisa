import Link from "next/link";
import { MODULES, MODULE_GROUPS } from "@/lib/modules";

export default function Home() {
  return (
    <div className="mx-auto max-w-6xl px-4">
      {/* Hero: serif thesis over warm white, ribbon eyebrow (guide pp. 30, 34) */}
      <section className="py-12 sm:py-16">
        <p className="ribbon">Farmer School of Business</p>
        <h1 className="mt-5 max-w-3xl text-4xl leading-tight sm:text-5xl">
          AI tools for your coursework, your job search, and beyond.
        </h1>
        <p className="mt-4 max-w-2xl text-lg leading-relaxed">
          ChatISA gives Miami students sponsored access to leading AI models,
          both commercial and open-source, for coding, exam prep, projects, job
          applications, interview practice, and general help.
        </p>
      </section>

      {MODULE_GROUPS.map(({ group, heading }) => {
        const modules = MODULES.filter((m) => m.group === group);
        if (modules.length === 0) return null;
        return (
          <section
            key={group}
            aria-labelledby={`group-${group}`}
            className="pb-8"
          >
            <h2 id={`group-${group}`} className="mb-3 text-2xl">
              {heading}
            </h2>
            <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {modules.map((mod) => (
                <li key={mod.slug}>
                  <Link
                    href={`/${mod.slug}`}
                    className="group block h-full rounded-card border border-medium-tan bg-paper p-5 shadow-card hover:border-miami-red"
                  >
                    <h3 className="text-xl text-ink group-hover:text-miami-red">
                      {mod.name}
                    </h3>
                    <p className="mt-2 text-sm leading-relaxed text-ink">
                      {mod.description}
                    </p>
                    <p className="mt-4 text-sm font-bold text-accent-red">
                      Open {mod.name}
                      <span aria-hidden="true"> →</span>
                    </p>
                  </Link>
                </li>
              ))}
            </ul>
          </section>
        );
      })}
    </div>
  );
}
