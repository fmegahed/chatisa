import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { MODULES, getModule } from "@/lib/modules";
import { ModulePlaceholder } from "@/components/ModulePlaceholder";

/**
 * Placeholder route for modules not yet rebuilt. A real static route
 * (e.g. app/coding-tutor/page.tsx) takes precedence over this dynamic
 * segment, so each slice replaces its placeholder simply by existing.
 */
export function generateStaticParams() {
  return MODULES.map((m) => ({ module: m.slug }));
}

export const dynamicParams = false;

export async function generateMetadata({
  params,
}: {
  params: Promise<{ module: string }>;
}): Promise<Metadata> {
  const mod = getModule((await params).module);
  return { title: mod ? mod.name : "Not found" };
}

export default async function ModulePage({
  params,
}: {
  params: Promise<{ module: string }>;
}) {
  const mod = getModule((await params).module);
  if (!mod) notFound();
  return <ModulePlaceholder module={mod} />;
}
