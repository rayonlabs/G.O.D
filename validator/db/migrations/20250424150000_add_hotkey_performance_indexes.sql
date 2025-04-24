-- migrate:up
BEGIN;

CREATE INDEX IF NOT EXISTS idx_submissions_hotkey
ON submissions_table(hotkey);

CREATE INDEX IF NOT EXISTS idx_submissions_hotkey_created
ON submissions_table(hotkey, created_on DESC);

CREATE INDEX IF NOT EXISTS idx_task_nodes_taskid_hotkey
ON task_nodes_table(task_id, hotkey);

CREATE INDEX IF NOT EXISTS idx_submissions_taskid_hotkey
ON submissions_table(task_id, hotkey);

CREATE INDEX IF NOT EXISTS idx_offer_responses_taskid_hotkey
ON offer_responses_table(task_id, hotkey);

CREATE INDEX IF NOT EXISTS idx_instruct_text_tasks_taskid
ON instruct_text_tasks_table(task_id);

CREATE INDEX IF NOT EXISTS idx_image_tasks_taskid
ON image_tasks_table(task_id);

CREATE INDEX IF NOT EXISTS idx_dpo_tasks_taskid
ON dpo_tasks_table(task_id);

COMMIT;

-- migrate:down
BEGIN;

DROP INDEX IF EXISTS idx_submissions_hotkey;
DROP INDEX IF EXISTS idx_submissions_hotkey_created;
DROP INDEX IF EXISTS idx_task_nodes_taskid_hotkey;
DROP INDEX IF EXISTS idx_submissions_taskid_hotkey;
DROP INDEX IF EXISTS idx_offer_responses_taskid_hotkey;
DROP INDEX IF EXISTS idx_instruct_text_tasks_taskid;
DROP INDEX IF EXISTS idx_image_tasks_taskid;
DROP INDEX IF EXISTS idx_dpo_tasks_taskid;

COMMIT;

