CREATE TABLE `scout_postings` (
	`id` text PRIMARY KEY NOT NULL,
	`source` text NOT NULL,
	`external_id` text NOT NULL,
	`fingerprint` text NOT NULL,
	`title` text NOT NULL,
	`company` text NOT NULL,
	`location_city` text,
	`location_state` text,
	`remote` integer DEFAULT false NOT NULL,
	`category` text NOT NULL,
	`apply_url` text NOT NULL,
	`description` text NOT NULL,
	`posted_at` text,
	`harvested_at` text NOT NULL,
	`last_seen_at` text NOT NULL,
	`skills_json` text DEFAULT '[]' NOT NULL,
	`taxonomy_version` integer NOT NULL,
	`active` integer DEFAULT true NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `scout_postings_source_external` ON `scout_postings` (`source`,`external_id`);--> statement-breakpoint
CREATE INDEX `scout_postings_active_category` ON `scout_postings` (`active`,`category`);--> statement-breakpoint
CREATE TABLE `scout_runs` (
	`id` text PRIMARY KEY NOT NULL,
	`started_at` text NOT NULL,
	`finished_at` text,
	`status` text NOT NULL,
	`trigger` text NOT NULL,
	`jsearch_requests` integer DEFAULT 0 NOT NULL,
	`jsearch_found` integer DEFAULT 0 NOT NULL,
	`usajobs_requests` integer DEFAULT 0 NOT NULL,
	`usajobs_found` integer DEFAULT 0 NOT NULL,
	`deduped_count` integer DEFAULT 0 NOT NULL,
	`tagged_count` integer DEFAULT 0 NOT NULL,
	`cost_usd` real DEFAULT 0 NOT NULL,
	`source_errors_json` text DEFAULT '{}' NOT NULL,
	`error` text
);
--> statement-breakpoint
CREATE INDEX `scout_runs_started` ON `scout_runs` (`started_at`);