# Trainer Logs Setup

## Architecture
- **Validator Server**: Hosts the observability stack (Grafana, Loki, Prometheus)
- **Trainer Nodes**: Ship logs to the validator's Loki instance

## Required Environment Variables

### On Validator (.vali.env)
Add these to your existing `.vali.env`:
```bash
# Training Logs Observability
OBSERVABILITY_DOMAIN=your-validator-domain.com
GRAFANA_TRAINING_PASSWORD=secure-password-here
LOKI_PASSWORD=loki-password-here
GRAFANA_ANONYMOUS_ENABLED=true  # Optional: allow public viewing
```

### On Trainer Nodes (.trainer.env)
Add these to your existing `.trainer.env`:
```bash
# Log Shipping to Validator
LOKI_ENDPOINT=https://your-validator-domain.com:3101
LOKI_PASSWORD=loki-password-here  # Same as in .vali.env
```

## Deployment

### Step 1: Deploy on Validator
```bash
# On your validator server
cd G.O.D
git pull

# Deploy observability stack
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
# On each trainer node
cd G.O.D
git pull

# Deploy log shipping
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