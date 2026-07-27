CREATE TABLE `interview_turns` (
	`id` text PRIMARY KEY NOT NULL,
	`interview_id` text NOT NULL,
	`ordinal` integer NOT NULL,
	`question` text NOT NULL,
	`topic` text,
	`answer_text` text,
	`answer_source` text,
	`answer_seconds` integer,
	`criteria_json` text,
	`strength` text,
	`improvement` text,
	`asked_at` text NOT NULL,
	`answered_at` text,
	FOREIGN KEY (`interview_id`) REFERENCES `interviews`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE UNIQUE INDEX `interview_turns_ordinal` ON `interview_turns` (`interview_id`,`ordinal`);--> statement-breakpoint
CREATE TABLE `interviews` (
	`id` text PRIMARY KEY NOT NULL,
	`user_email` text NOT NULL,
	`model_id` text NOT NULL,
	`interview_type` text NOT NULL,
	`status` text NOT NULL,
	`job_title` text NOT NULL,
	`role_brief` text,
	`candidate_brief` text,
	`grade_level` text,
	`major` text,
	`planned_questions` integer NOT NULL,
	`asked_count` integer DEFAULT 0 NOT NULL,
	`summary_json` text,
	`created_at` text NOT NULL,
	`completed_at` text,
	FOREIGN KEY (`user_email`) REFERENCES `users`(`email`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `interviews_user_created` ON `interviews` (`user_email`,`created_at`);