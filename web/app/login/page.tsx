import Image from "next/image";
import { redirect } from "next/navigation";
import type { Metadata } from "next";
import { AuthError } from "next-auth";
import { auth, signIn } from "@/lib/auth";
import { isTestModeEnabled } from "@/lib/auth/domain";

export const metadata: Metadata = { title: "Sign in" };

const ERROR_MESSAGES: Record<string, string> = {
  AccessDenied:
    "That Google account can't be used here. Sign in with your @miamioh.edu account.",
  CredentialsSignin:
    "That email can't be used here. Use an @miamioh.edu address.",
  Configuration:
    "Sign-in isn't configured on this server yet. Contact the ChatISA maintainers.",
};

export default async function LoginPage({
  searchParams,
}: {
  searchParams: Promise<{ error?: string }>;
}) {
  const session = await auth();
  if (session?.user?.email) redirect("/");

  const { error } = await searchParams;
  const errorMessage = error
    ? (ERROR_MESSAGES[error] ??
      "Sign-in didn't complete. Try again, and use your @miamioh.edu account.")
    : null;
  const testMode = isTestModeEnabled(process.env);

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
            // h-auto AND w-auto: next/image warns when CSS touches one
            // dimension without the other (console notice, 2026-07-29);
            // the pair keeps the mark's aspect under the global reset.
            className="mx-auto h-auto w-auto"
          />
          <h1 className="mt-6 text-center text-3xl">Sign in to ChatISA</h1>
          <p className="mt-2 text-center">
            Free AI tools for Miami students: coding help, project coaching,
            exam prep, interview practice, and more.
          </p>

          {errorMessage ? (
            <div
              role="alert"
              className="mt-6 rounded-card border-2 border-miami-red bg-warm-white p-4"
            >
              <p className="font-bold text-miami-red">Sign-in problem</p>
              <p className="mt-1 text-ink">{errorMessage}</p>
            </div>
          ) : null}

          <form
            className="mt-6"
            action={async () => {
              "use server";
              await signIn("google", { redirectTo: "/" });
            }}
          >
            <button
              type="submit"
              className="w-full rounded-card bg-miami-red px-4 py-3 font-bold text-paper hover:bg-accent-red"
            >
              Sign in with your Miami Google account
            </button>
          </form>
          <p className="mt-3 text-center text-sm text-dark-tan">
            Only @miamioh.edu accounts can sign in.
          </p>

          {testMode ? (
            <form
              className="mt-8 border-t border-medium-tan pt-6"
              action={async (formData: FormData) => {
                "use server";
                try {
                  await signIn("test-login", {
                    email: String(formData.get("email") ?? ""),
                    redirectTo: "/",
                  });
                } catch (err) {
                  if (err instanceof AuthError) {
                    redirect(`/login?error=${err.type}`);
                  }
                  throw err;
                }
              }}
            >
              <p className="font-bold">Test login (non-production only)</p>
              <label htmlFor="test-email" className="mt-2 block text-sm">
                Email address
              </label>
              <input
                id="test-email"
                name="email"
                type="email"
                required
                autoComplete="email"
                className="mt-1 w-full rounded-card border border-medium-tan bg-paper px-3 py-2"
              />
              <button
                type="submit"
                className="mt-3 w-full rounded-card border border-medium-tan px-4 py-2 font-bold hover:border-miami-red"
              >
                Sign in as test user
              </button>
            </form>
          ) : null}
        </div>
        <p className="mt-6 text-center text-sm text-dark-tan">
          Maintained by the Farmer School of Business. Educational use only.
        </p>
      </div>
    </main>
  );
}
