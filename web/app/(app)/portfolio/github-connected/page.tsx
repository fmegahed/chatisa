import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { auth } from "@/lib/auth";
import { GithubConnected } from "@/components/scout/GithubConnected";

export const metadata: Metadata = { title: "GitHub connected" };

/**
 * Landing page for the GitHub OAuth popup (v6.3.0). The token arrives in
 * the URL fragment, which only the client component can read; this server
 * shell just gates on a session like the rest of the module.
 */
export default async function GithubConnectedPage() {
  const session = await auth();
  if (!session?.user?.email) redirect("/login");
  return (
    <div className="mx-auto max-w-xl px-4 py-16">
      <GithubConnected />
    </div>
  );
}
