# How to go through validator logs

We use opentelemetry to instrument the validator and get logs to grafana with some context fields like task_id, miner_hotkey etc.

The grafana will be available at `http://localhost:3000` and the default username is `admin` and password is generated during startup and stored in the .vali.env.

In grafana go to the dashboard `Validator Logs` to see the logs. You can filter by task_id and miner_hotkey in the text boxes at the top of the page. Or you can use the query builder to build a query if you are brave.
