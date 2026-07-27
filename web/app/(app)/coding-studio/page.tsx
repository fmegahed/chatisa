import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { auth } from "@/lib/auth";
import { SandboxClient } from "@/components/sandbox/SandboxClient";
import { buildModelOptions, getPageModels } from "@/lib/config/models";
import { filterAvailableModels } from "@/lib/providers";

export const metadata: Metadata = { title: "Coding Studio" };

export default async function AiSandboxPage() {
  const session = await auth();
  if (!session?.user?.email) redirect("/login");

  // The side chat offers the same models as the Coding Companion.
  const available = filterAvailableModels(getPageModels("sandbox_chat"));
  const { options, defaultModelId } = buildModelOptions(
    "sandbox_chat",
    available,
  );

  return (
    <SandboxClient
      models={options}
      defaultModelId={defaultModelId}
      userEmail={session.user.email}
    />
  );
}
