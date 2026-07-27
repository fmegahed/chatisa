CREATE TABLE `job_applications` (
	`id` text PRIMARY KEY NOT NULL,
	`user_email` text NOT NULL,
	`company` text NOT NULL,
	`position_title` text NOT NULL,
	`job_url` text,
	`description_source` text DEFAULT 'none' NOT NULL,
	`posting_text` text,
	`role_brief` text,
	`candidate_brief` text,
	`resume_text` text,
	`resume_filename` text,
	`resume_purged_at` text,
	`created_at` text NOT NULL,
	`updated_at` text NOT NULL,
	FOREIGN KEY (`user_email`) REFERENCES `users`(`email`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `job_applications_user` ON `job_applications` (`user_email`,`created_at`);--> statement-breakpoint
CREATE TABLE `tailored_documents` (
	`id` text PRIMARY KEY NOT NULL,
	`application_id` text NOT NULL,
	`user_email` text NOT NULL,
	`kind` text NOT NULL,
	`template` integer DEFAULT 2 NOT NULL,
	`model_id` text NOT NULL,
	`content_json` text NOT NULL,
	`ungrounded_json` text,
	`reviewed_at` text,
	`created_at` text NOT NULL,
	`updated_at` text NOT NULL,
	FOREIGN KEY (`application_id`) REFERENCES `job_applications`(`id`) ON UPDATE no action ON DELETE cascade,
	FOREIGN KEY (`user_email`) REFERENCES `users`(`email`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `tailored_documents_application` ON `tailored_documents` (`application_id`,`kind`);--> statement-breakpoint
ALTER TABLE `interviews` ADD `application_id` text REFERENCES job_applications(id);