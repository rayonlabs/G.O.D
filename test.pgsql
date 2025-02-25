-- Get all-0-score tasks in last 12 hours
WITH tasks_last_12h AS (
    SELECT *
    FROM tasks
    WHERE termination_at >= NOW() - INTERVAL '12 HOURS'
),

zero_score_tasks AS (
    SELECT DISTINCT t.task_id
    FROM tasks_last_12h t
    JOIN task_nodes tn ON t.task_id = tn.task_id
    WHERE t.n_eval_attempts = 3
    GROUP BY t.task_id
    HAVING MAX(tn.quality_score) = 0 AND MIN(tn.quality_score) = 0
),

-- Select miners submissions
tasks_submissions_last_12h AS (
SELECT tn.task_id, tn.hotkey, tn.quality_score, s.repo
FROM task_nodes tn
LEFT JOIN submissions s ON tn.task_id = s.task_id AND tn.hotkey = s.hotkey
WHERE tn.task_id IN (SELECT task_id FROM zero_score_tasks)
)

-- Basic stats
-- SELECT COUNT(*) as total_tasks_12h
--     , (SELECT COUNT(*) FROM zero_score_tasks) as zero_score_tasks_count
--     , ROUND(
--         (SELECT COUNT(*) FROM zero_score_tasks)::numeric /
--         NULLIF(COUNT(*), 0) * 100::numeric
--     , 2) as zero_score_percentage
-- FROM tasks_last_12h;

-- Detailed results in regular table format
-- SELECT t.*
-- FROM tasks_last_12h t
-- JOIN zero_score_tasks zst ON t.task_id = zst.task_id
-- ORDER BY t.created_at DESC;

-- Selected fields in regular table format
-- SELECT
--     t.task_id,
--     t.model_id,
--     t.created_at,
--     t.ds_id,
--     t.field_instruction,
--     t.field_input,
--     t.field_output,
--     t.status,
--     t.test_data,
--     t.training_data,
--     t.is_organic
-- FROM tasks_last_12h t
-- JOIN zero_score_tasks zst ON t.task_id = zst.task_id
-- ORDER BY t.created_at DESC;

-- Join with tasks
SELECT ts.*, t.model_id, t.ds_id, t.field_instruction, t.field_input, t.field_output, t.status, t.test_data, t.training_data, t.is_organic, t.trained_model_repository
FROM tasks_submissions_last_12h ts
JOIN tasks t ON ts.task_id = t.task_id
ORDER BY ts.task_id;


--------------------------------
-- Get successful tasks, most recent
select * from tasks where status = 'success' order by termination_at desc;

-- Re-evaluate tasks
update tasks set status = 'training' where task_id in ('<TASK_UUID>');

-- Get training tasks, most recent
select * from tasks
where status = 'training'
order by termination_at desc;

-- Move finished training tasks to evaluation
update tasks set termination_at = NOW() where status = 'training' and task_id in ('<TASK_UUID>');

-- Get task details
select * from tasks
where task_id = '<TASK_UUID>';

-- Get task breakdown
SELECT s.repo, tn.*, t.*, tn.hotkey as submitted_repo
FROM tasks t
JOIN task_nodes tn ON t.task_id = tn.task_id
LEFT JOIN submissions s ON tn.task_id = s.task_id AND tn.hotkey = s.hotkey
WHERE t.task_id = '<TASK_UUID>'
ORDER BY tn.quality_score DESC;
