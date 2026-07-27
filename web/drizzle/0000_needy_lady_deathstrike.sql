CREATE TABLE `users` (
	`email` text PRIMARY KEY NOT NULL,
	`name` text,
	`role` text DEFAULT 'student' NOT NULL,
	`first_seen_at` text NOT NULL,
	`last_seen_at` text NOT NULL
);
