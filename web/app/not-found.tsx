import Link from "next/link";
import { DocumentTitle } from "@/components/a11y/DocumentTitle";

export default function NotFound() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-16">
      {/* A specific title instead of the site default (WCAG 2.4.2).
          not-found.tsx cannot export metadata, and a rendered <title> loses
          to the root layout's default. */}
      <DocumentTitle title="Page not found · ChatISA" />
      <p className="ribbon">Page not found</p>
      <h1 className="mt-5 text-4xl">There&apos;s nothing at this address.</h1>
      <p className="mt-4 text-lg">
        The page may have moved while ChatISA is being rebuilt.
      </p>
      <p className="mt-6">
        <Link
          href="/"
          className="font-bold text-accent-red underline underline-offset-2"
        >
          Go to the ChatISA home page
        </Link>
      </p>
    </div>
  );
}
