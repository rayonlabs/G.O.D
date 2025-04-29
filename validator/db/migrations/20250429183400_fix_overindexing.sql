-- migrate:up
-- Remove duplicate indices from dpo_tasks
DROP INDEX IF EXISTS idx_dpo_task_id;
DROP INDEX IF EXISTS idx_dpo_tasks_taskid;

-- Remove duplicate indices from image_tasks
DROP INDEX IF EXISTS idx_image_task_id;
DROP INDEX IF EXISTS idx_image_tasks_taskid;

-- Remove duplicate indices from instruct_text_tasks
DROP INDEX IF EXISTS idx_instruct_text_tasks_taskid;
DROP INDEX IF EXISTS idx_instruct_text_task_id;

-- Remove duplicate indices from nodes
DROP INDEX IF EXISTS idx_nodes_hotkey;

-- Remove duplicate indices from submissions
DROP INDEX IF EXISTS idx_submissions_taskid_hotkey;
DROP INDEX IF EXISTS idx_submissions_hotkey;

-- Remove duplicate indices from task_nodes
DROP INDEX IF EXISTS idx_task_nodes_taskid_hotkey;
DROP INDEX IF EXISTS idx_task_nodes_hotkey_netuid;

-- Remove duplicate indices from tasks
DROP INDEX IF EXISTS idx_tasks_created_at;

-- migrate:down
CREATE INDEX IF NOT EXISTS idx_dpo_task_id ON dpo_tasks(task_id);
CREATE INDEX IF NOT EXISTS idx_image_task_id ON image_tasks(task_id);
CREATE INDEX IF NOT EXISTS idx_instruct_text_task_id ON instruct_text_tasks(task_id);
CREATE INDEX IF NOT EXISTS idx_nodes_hotkey ON nodes(hotkey);
CREATE INDEX IF NOT EXISTS idx_submissions_taskid_hotkey ON submissions(task_id, hotkey);
CREATE INDEX IF NOT EXISTS idx_submissions_hotkey ON submissions(hotkey);
CREATE INDEX IF NOT EXISTS idx_task_nodes_taskid_hotkey ON task_nodes(task_id, hotkey);
CREATE INDEX IF NOT EXISTS idx_task_nodes_hotkey_netuid ON task_nodes(hotkey, netuid);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);
