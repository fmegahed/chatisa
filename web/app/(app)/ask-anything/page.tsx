import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { auth } from "@/lib/auth";
import { AskAnythingClient } from "@/components/ask/AskAnythingClient";
import { CHAT_MODULES } from "@/lib/chat/config";
import { buildModelOptions, getPageModels } from "@/lib/config/models";
import { filterAvailableModels } from "@/lib/providers";

export const metadata: Metadata = { title: "Ask Anything" };

export default async function AskAnythingPage() {
  const session = await auth();
  if (!session?.user?.email) redirect("/login");

  const mod = CHAT_MODULES.ask_anything;
  const available = filterAvailableModels(getPageModels(mod.key));
  const { options, defaultModelId } = buildModelOptions(mod.key, available);

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <p className="ribbon">Module</p>
      <h1 className="mt-5 text-4xl">{mod.name}</h1>
      <p className="mt-3 max-w-2xl text-lg leading-relaxed">
        A general assistant with the frontier model of your choice. Your chats
        are saved on this device only, never on a server.
      </p>

      {options.length === 0 ? (
        <div
          role="status"
          className="mt-8 rounded-card border-2 border-miami-red bg-paper p-5"
        >
          <h2 className="font-bold text-miami-red">No models are available</h2>
          <p className="mt-1">
            This server has no AI provider configured yet. Contact the ChatISA
            maintainers.
          </p>
        </div>
      ) : (
        <div className="mt-8">
          <AskAnythingClient models={options} defaultModelId={defaultModelId} />
        </div>
      )}
    </div>
  );
}
