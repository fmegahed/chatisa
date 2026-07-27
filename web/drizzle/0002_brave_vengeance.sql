CREATE TABLE `exam_answers` (
	`id` text PRIMARY KEY NOT NULL,
	`exam_id` text NOT NULL,
	`question_id` text NOT NULL,
	`selected_index` integer,
	`response_text` text,
	`confidence` text,
	`graded_by` text,
	`grader_model_id` text,
	`is_correct` integer,
	`points_awarded` real,
	`criteria_json` text,
	`feedback` text,
	`created_at` text NOT NULL,
	`graded_at` text,
	FOREIGN KEY (`exam_id`) REFERENCES `exams`(`id`) ON UPDATE no action ON DELETE cascade,
	FOREIGN KEY (`question_id`) REFERENCES `exam_questions`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE UNIQUE INDEX `exam_answers_question` ON `exam_answers` (`question_id`);--> statement-breakpoint
CREATE TABLE `exam_document_pages` (
	`id` text PRIMARY KEY NOT NULL,
	`document_id` text NOT NULL,
	`page_number` integer NOT NULL,
	`text` text NOT NULL,
	`char_count` integer NOT NULL,
	`source` text NOT NULL,
	FOREIGN KEY (`document_id`) REFERENCES `exam_documents`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE UNIQUE INDEX `exam_document_pages_doc_page` ON `exam_document_pages` (`document_id`,`page_number`);--> statement-breakpoint
CREATE TABLE `exam_documents` (
	`id` text PRIMARY KEY NOT NULL,
	`user_email` text NOT NULL,
	`filename` text NOT NULL,
	`size_bytes` integer NOT NULL,
	`page_count` integer NOT NULL,
	`text_page_count` integer NOT NULL,
	`vision_page_count` integer DEFAULT 0 NOT NULL,
	`char_count` integer NOT NULL,
	`classification` text NOT NULL,
	`warnings_json` text DEFAULT '[]' NOT NULL,
	`text_purged_at` text,
	`created_at` text NOT NULL,
	FOREIGN KEY (`user_email`) REFERENCES `users`(`email`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `exam_documents_user_created` ON `exam_documents` (`user_email`,`created_at`);--> statement-breakpoint
CREATE TABLE `exam_questions` (
	`id` text PRIMARY KEY NOT NULL,
	`exam_id` text NOT NULL,
	`position` integer NOT NULL,
	`type` text NOT NULL,
	`stem` text NOT NULL,
	`options_json` text,
	`correct_index` integer,
	`model_answer` text NOT NULL,
	`rubric_json` text NOT NULL,
	`explanation` text NOT NULL,
	`topic` text NOT NULL,
	`bloom` text NOT NULL,
	`source_quote` text NOT NULL,
	`source_page` integer NOT NULL,
	`grounding_status` text NOT NULL,
	`points_possible` integer NOT NULL,
	FOREIGN KEY (`exam_id`) REFERENCES `exams`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE UNIQUE INDEX `exam_questions_exam_position` ON `exam_questions` (`exam_id`,`position`);--> statement-breakpoint
CREATE TABLE `exams` (
	`id` text PRIMARY KEY NOT NULL,
	`user_email` text NOT NULL,
	`document_id` text NOT NULL,
	`model_id` text NOT NULL,
	`status` text NOT NULL,
	`failure_reason` text,
	`exam_mode` text NOT NULL,
	`question_type` text NOT NULL,
	`requested_count` integer NOT NULL,
	`delivered_count` integer DEFAULT 0 NOT NULL,
	`dropped_count` integer DEFAULT 0 NOT NULL,
	`scope_from_page` integer NOT NULL,
	`scope_to_page` integer NOT NULL,
	`coverage_json` text DEFAULT '{}' NOT NULL,
	`current_position` integer DEFAULT 0 NOT NULL,
	`created_at` text NOT NULL,
	`updated_at` text NOT NULL,
	`completed_at` text,
	FOREIGN KEY (`user_email`) REFERENCES `users`(`email`) ON UPDATE no action ON DELETE cascade,
	FOREIGN KEY (`document_id`) REFERENCES `exam_documents`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `exams_user_updated` ON `exams` (`user_email`,`updated_at`);--> statement-breakpoint
CREATE INDEX `exams_document` ON `exams` (`document_id`);