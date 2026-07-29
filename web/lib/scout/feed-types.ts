/**
 * Client-side shapes of the feed index (GET /api/scout/feed?shape=index).
 * Shared by every Job Scout tab: the Jobs tab renders it, the Profile tab
 * derives demand analytics from it, the Saved tab joins against it.
 */

export interface FeedPosting {
  id: string;
  source: string;
  title: string;
  company: string;
  locationCity: string | null;
  locationState: string | null;
  remote: boolean;
  category: "fulltime" | "internship" | "federal";
  applyUrl: string;
  postedAt: string | null;
  skillsJson: string;
  visaSponsorship: "sponsors" | "no_sponsorship" | "unknown";
}

export interface FeedFreshness {
  updatedAt: string | null;
  totalActive: number;
  sourceErrors: { activejobs?: string; usajobs?: string; tagging?: string };
}

export interface FeedIndex {
  postings: FeedPosting[];
  freshness: FeedFreshness;
}

/** Top skills across the active feed, for the Profile tab's demand panel. */
export function demandRanking(
  postings: FeedPosting[],
  limit = 10,
): { skillId: string; count: number }[] {
  const counts = new Map<string, number>();
  for (const p of postings) {
    try {
      for (const s of JSON.parse(p.skillsJson) as { skillId: string }[]) {
        counts.set(s.skillId, (counts.get(s.skillId) ?? 0) + 1);
      }
    } catch {
      // A malformed row contributes nothing rather than crashing analytics.
    }
  }
  return [...counts.entries()]
    .map(([skillId, count]) => ({ skillId, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, limit);
}
