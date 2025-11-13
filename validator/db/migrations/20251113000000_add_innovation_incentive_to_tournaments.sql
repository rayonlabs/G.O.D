-- migrate:up

ALTER TABLE tournaments
ADD COLUMN innovation_incentive DOUBLE PRECISION;

-- migrate:down

ALTER TABLE tournaments
DROP COLUMN IF EXISTS innovation_incentive;
