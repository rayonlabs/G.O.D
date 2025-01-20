-- migrate:up
ALTER TABLE tasks ADD COLUMN test_data_num_rows INTEGER;
ALTER TABLE tasks ADD COLUMN synthetic_data_num_rows INTEGER;
ALTER TABLE tasks ADD COLUMN training_data_num_rows INTEGER;

-- migrate:down
ALTER TABLE tasks DROP COLUMN test_data_num_rows;
ALTER TABLE tasks DROP COLUMN synthetic_data_num_rows;
ALTER TABLE tasks DROP COLUMN training_data_num_rows;