import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { getScoutPosting } from "@/lib/db";

/**
 * Full posting detail (description included), fetched on first card expand.
 * Public employer content, so any signed-in student may read any posting;
 * nothing records who read what.
 */
export async function GET(
  _req: Request,
  ctx: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  if (!session?.user?.email) {
    return NextResponse.json({ error: "Sign in required." }, { status: 401 });
  }
  const { id } = await ctx.params;
  const posting = getScoutPosting(id);
  if (!posting) {
    return NextResponse.json(
      { error: "This posting is no longer in the feed." },
      { status: 404 },
    );
  }
  return NextResponse.json({ posting });
}
