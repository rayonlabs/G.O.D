-- migrate:up
BEGIN;

CREATE INDEX IF NOT EXISTS idx_submissions_hotkey
ON submissions(hotkey);

CREATE INDEX IF NOT EXISTS idx_submissions_hotkey_created
ON submissions(hotkey, created_on DESC);

CREATE INDEX IF NOT EXISTS idx_task_nodes_taskid_hotkey
ON task_nodes(task_id, hotkey);

CREATE INDEX IF NOT EXISTS idx_submissions_taskid_hotkey
ON submissions(task_id, hotkey);

CREATE INDEX IF NOT EXISTS idx_offer_responses_taskid_hotkey
ON offer_responses(task_id, hotkey);

CREATE INDEX IF NOT EXISTS idx_instruct_text_tasks_taskid
ON instruct_text_tasks(task_id);

CREATE INDEX IF NOT EXISTS idx_image_tasks_taskid
ON image_tasks(task_id);

CREATE INDEX IF NOT EXISTS idx_dpo_tasks_taskid
ON dpo_tasks(task_id);

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

