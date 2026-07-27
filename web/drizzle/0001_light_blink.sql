CREATE TABLE `conversations` (
	`id` text PRIMARY KEY NOT NULL,
	`user_email` text NOT NULL,
	`module` text NOT NULL,
	`title` text NOT NULL,
	`model_id` text NOT NULL,
	`created_at` text NOT NULL,
	`updated_at` text NOT NULL,
	FOREIGN KEY (`user_email`) REFERENCES `users`(`email`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `conversations_user_updated` ON `conversations` (`user_email`,`updated_at`);--> statement-breakpoint
CREATE TABLE `messages` (
	`id` text PRIMARY KEY NOT NULL,
	`conversation_id` text NOT NULL,
	`role` text NOT NULL,
	`content` text NOT NULL,
	`model_id` text,
	`input_tokens` integer,
	`output_tokens` integer,
	`cost_usd` real,
	`created_at` text NOT NULL,
	FOREIGN KEY (`conversation_id`) REFERENCES `conversations`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `messages_conversation_created` ON `messages` (`conversation_id`,`created_at`);--> statement-breakpoint
CREATE TABLE `usage_events` (
	`id` text PRIMARY KEY NOT NULL,
	`created_at` text NOT NULL,
	`user_email` text,
	`module` text NOT NULL,
	`event_type` text NOT NULL,
	`model_id` text,
	`provider` text,
	`input_tokens` integer,
	`output_tokens` integer,
	`cost_usd` real,
	`latency_ms` integer,
	`prompt_chars` integer,
	`response_chars` integer,
	`outcome` text
);
--> statement-breakpoint
CREATE INDEX `usage_events_created` ON `usage_events` (`created_at`);