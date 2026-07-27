import Image from "next/image";
import { redirect } from "next/navigation";
import type { Metadata } from "next";
import { AuthError } from "next-auth";
import { auth, signIn } from "@/lib/auth";
import { guestPassesEnabled } from "@/lib/auth/guest";

export const metadata: Metadata = { title: "Collaborator access" };

/**
 * Landing page for guest magic links (chatisa/guest?pass=...). One explicit
 * button rather than an auto-submit: the visitor sees what they are joining,
 * and an invalid or expired link gets a clear explanation instead of a
 * mysterious bounce.
 */
export default async function GuestPage({
  searchParams,
}: {
  searchParams: Promise<{ pass?: string; error?: string }>;
}) {
  const session = await auth();
  if (session?.user?.email) redirect("/");

  const { pass, error } = await searchParams;
  // Wall-clock is per-request here, not per-re-render: this is a dynamic
  // server component (the auth() call above opts out of static rendering),
  // so expiry is evaluated freshly on every visit.
  // eslint-disable-next-line react-hooks/purity
  const enabled = guestPassesEnabled(process.env, Date.now());

  return (
    <main id="main" tabIndex={-1} className="flex flex-1 items-center px-4 py-12">
      <div className="mx-auto w-full max-w-md">
        <div className="rounded-card border border-medium-tan bg-paper p-8 shadow-card">
          <Image
            src="/brand/logo-vertical-stacked.png"
            alt="Miami University"
            width={164}
            height={120}
            priority
            className="mx-auto h-auto"
          />
          <h1 className="mt-6 text-center text-3xl">
            You&apos;re invited to try ChatISA
          </h1>
          <p className="mt-2 text-center">
            A guest pass gives you full access to Miami University&apos;s AI
            tools for students: coding help, data analysis, research search,
            exam prep, and more.
          </p>

          {error ? (
            <div
              role="alert"
              className="mt-6 rounded-card border-2 border-miami-red bg-warm-white p-4"
            >
              <p className="font-bold text-miami-red">
                This invite link didn&apos;t work
              </p>
              <p className="mt-1 text-ink">
                The pass may have expired or been revoked. Ask the person who
                invited you for a fresh link.
              </p>
            </div>
          ) : null}

          {!enabled ? (
            <p className="mt-6 text-center text-dark-tan">
              Guest access is not open right now. If you were sent a link, ask
              your contact at Miami for a current one.
            </p>
          ) : !pass ? (
            <p className="mt-6 text-center text-dark-tan">
              This page needs the full invite link you were sent (it ends with
              your personal pass code). Open the link exactly as you received
              it.
            </p>
          ) : (
            <form
              className="mt-6"
              action={async () => {
                "use server";
                try {
                  await signIn("guest-pass", { pass, redirectTo: "/" });
                } catch (err) {
                  if (err instanceof AuthError) {
                    redirect("/guest?error=1");
                  }
                  throw err;
                }
              }}
            >
              <button
                type="submit"
                className="w-full rounded-card bg-miami-red px-4 py-3 font-bold text-paper hover:bg-accent-red"
              >
                Enter ChatISA as a guest
              </button>
            </form>
          )}
        </div>
        <p className="mt-6 text-center text-sm text-dark-tan">
          Guest sessions are for evaluation. Conversations stay in your own
          browser; usage is metered like any student account.
        </p>
      </div>
    </main>
  );
}
