CREATE TABLE `deliverables` (
	`id` text PRIMARY KEY NOT NULL,
	`project_id` text NOT NULL,
	`coach_type` text NOT NULL,
	`content_json` text DEFAULT '{}' NOT NULL,
	`transcript_json` text DEFAULT '[]' NOT NULL,
	`last_updated_by` text,
	`updated_at` text NOT NULL,
	FOREIGN KEY (`project_id`) REFERENCES `projects`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE UNIQUE INDEX `deliverables_project_coach` ON `deliverables` (`project_id`,`coach_type`);--> statement-breakpoint
CREATE TABLE `project_members` (
	`id` text PRIMARY KEY NOT NULL,
	`project_id` text NOT NULL,
	`email` text NOT NULL,
	`name` text,
	`role` text NOT NULL,
	`created_at` text NOT NULL,
	FOREIGN KEY (`project_id`) REFERENCES `projects`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE UNIQUE INDEX `project_members_project_email` ON `project_members` (`project_id`,`email`);--> statement-breakpoint
CREATE INDEX `project_members_email` ON `project_members` (`email`);--> statement-breakpoint
CREATE TABLE `projects` (
	`id` text PRIMARY KEY NOT NULL,
	`course_code` text NOT NULL,
	`name` text NOT NULL,
	`organization` text DEFAULT '' NOT NULL,
	`owner_email` text NOT NULL,
	`coach_types_json` text DEFAULT '[]' NOT NULL,
	`created_at` text NOT NULL,
	`updated_at` text NOT NULL,
	FOREIGN KEY (`owner_email`) REFERENCES `users`(`email`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `projects_owner_updated` ON `projects` (`owner_email`,`updated_at`);