import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { auth } from "@/lib/auth";
import { Chat } from "@/components/chat/Chat";
import { CHAT_MODULES } from "@/lib/chat/config";
import { buildModelOptions, getPageModels } from "@/lib/config/models";
import { filterAvailableModels } from "@/lib/providers";

export const metadata: Metadata = { title: "Coding Tutor" };

export default async function CodingCompanionPage() {
  const session = await auth();
  if (!session?.user?.email) redirect("/login");

  const mod = CHAT_MODULES.coding_companion;
  const available = filterAvailableModels(getPageModels(mod.key));
  const { options, defaultModelId } = buildModelOptions(mod.key, available);

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <p className="ribbon">Module</p>
      <h1 className="mt-5 text-4xl">{mod.name}</h1>
      <p className="mt-3 max-w-2xl text-lg leading-relaxed">
        Ask about code or analytics concepts. Answers explain the reasoning and
        show examples in both R and Python.
      </p>

      {options.length === 0 ? (
        <div role="status" className="mt-8 rounded-card border-2 border-miami-red bg-paper p-5">
          <h2 className="font-bold text-miami-red">No models are available</h2>
          <p className="mt-1">
            This server has no AI provider configured yet. Contact the ChatISA
            maintainers.
          </p>
        </div>
      ) : (
        <div className="mt-8">
          <Chat
            moduleKey={mod.key}
            moduleName={mod.name}
            placeholder={mod.placeholder}
            models={options}
            defaultModelId={defaultModelId}
          />
        </div>
      )}
    </div>
  );
}
