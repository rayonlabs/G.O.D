# Trainer Logs Setup

## Architecture
- **Validator Server**: Hosts the observability stack (Grafana, Loki, Prometheus)
- **Trainer Nodes**: Ship logs to the validator's Loki instance

## Configuration

### On Validator (.vali.env)
No configuration required! The system auto-detects the server's IP and uses secure defaults.

Optional overrides you can add to `.vali.env`:
```bash
# All optional - only set if you want to override defaults
OBSERVABILITY_DOMAIN=custom-domain.com  # Auto-detected if not set
GRAFANA_TRAINING_PASSWORD=custom-password  # Default: changeme123
LOKI_PASSWORD=custom-loki-password  # Default: trainerlogs123
GRAFANA_ANONYMOUS_ENABLED=false  # Default: true (allow public viewing)
```

### On Trainer Nodes (.trainer.env)
Add your validator's IP to your existing `.trainer.env`:
```bash
# Required
VALIDATOR_IP=45.79.123.456  # Your validator's IP address
```

That's it! The system auto-generates everything else.

## Deployment

### Step 1: Deploy on Validator
```bash
task deploy-observability-server
```

This will:
- Auto-generate SSL certificates
- Create authentication files
- Start Grafana on port 3001
- Start Loki on port 3101
- Start Prometheus for metrics

### Step 2: Deploy on Each Trainer
```bash
task deploy-trainer-logs
```

This will:
- Start Vector log shipper
- Automatically collect logs from training containers
- Ship logs to validator's Loki instance

## Access Grafana

Navigate to: `https://your-validator-domain.com:3001`
- Username: `admin` (or value of GRAFANA_TRAINING_USERNAME)
- Password: (value from GRAFANA_TRAINING_PASSWORD in .vali.env)

## Commands

| Command | Run On | Description |
|---------|--------|-------------|
| `task deploy-observability-server` | Validator | Deploy Grafana/Loki/Prometheus |
| `task stop-observability-server` | Validator | Stop observability stack |
| `task deploy-trainer-logs` | Trainer | Deploy log shipping |
| `task stop-trainer-logs` | Trainer | Stop log shipping |
| `task logs-observability` | Validator | View observability logs |
| `task logs-trainer-shipper` | Trainer | View Vector logs |
| `task test-trainer-logs` | Trainer | Create test container |

## Testing

After deployment, test the setup:

```bash
# On trainer node
task test-trainer-logs

# Then check Grafana for logs with task_id=test-*
```

## Troubleshooting

### Logs not appearing
1. Check Vector is running: `docker ps | grep vector`
2. Check Vector logs: `task logs-trainer-shipper`
3. Verify LOKI_ENDPOINT is correct in .trainer.env
4. Verify LOKI_PASSWORD matches between .vali.env and .trainer.env

### Connection errors
1. Check firewall allows port 3101 on validator
2. Verify SSL certificate (self-signed by default)
3. Check authentication with: `curl -u trainer:password https://validator:3101/ready`

## Security Notes

- SSL certificates are auto-generated (self-signed)
- Replace with proper certificates for production
- Loki requires authentication (basic auth)
- Grafana can be public or private (GRAFANA_ANONYMOUS_ENABLED)

## What Gets Logged

Vector automatically collects logs from containers matching:
- `image-trainer-*`
- `text-trainer-*`
- `downloader-*`
- `hf-upload-*`

Container labels are preserved for filtering:
- task_id
- hotkey
- model
- trainer_type
- expected_repo
