import Image from "next/image";
import Link from "next/link";
import { ModuleNav } from "./ModuleNav";
import { signOut } from "@/lib/auth";

/**
 * App header. White surface so the full-color Beveled-M sits on an
 * approved background (guide p. 11: no white/reverse variant available).
 */
export function SiteHeader({
  user,
}: {
  user?: { email?: string | null; name?: string | null };
}) {
  return (
    <header className="border-b-4 border-miami-red bg-paper">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-3 px-4 py-3">
        <Link
          href="/"
          className="flex items-center gap-3"
          aria-label="ChatISA home"
        >
          <Image
            src="/brand/beveled-m.png"
            alt=""
            width={40}
            height={29}
            priority
          />
          <span className="font-display text-2xl font-semibold text-ink">
            ChatISA
          </span>
        </Link>
        <p className="ml-2 hidden border-l border-medium-tan pl-3 text-sm text-dark-tan md:block">
          AI tools for Miami University students
        </p>
        {user?.email ? (
          <div className="ml-auto flex items-center gap-3">
            <p className="text-sm text-dark-tan">
              <span className="sr-only">Signed in as </span>
              {user.email}
            </p>
            <form
              action={async () => {
                "use server";
                await signOut({ redirectTo: "/login" });
              }}
            >
              <button
                type="submit"
                className="rounded-card border border-medium-tan bg-paper px-3 py-1.5 text-sm font-bold text-ink hover:border-miami-red hover:text-miami-red"
              >
                Sign out
              </button>
            </form>
          </div>
        ) : null}
      </div>
      <div className="mx-auto max-w-6xl border-t border-light-tan">
        <ModuleNav />
      </div>
    </header>
  );
}
