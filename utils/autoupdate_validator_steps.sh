# Steps to take to update the validator automatically
# Change each time but take caution


. $HOME/.venv/bin/activate
pip install -e .

task validator

# Update observability server if it's running
if docker-compose -f docker-compose.observability-server.yml ps 2>/dev/null | grep -q "god-grafana-training"; then
    echo "Updating observability server..."
    
    # Check if dashboard config changed
    if git diff HEAD~1 HEAD --name-only 2>/dev/null | grep -q "grafana-training-dashboard.json"; then
        echo "Dashboard config changed, restarting Grafana..."
        docker-compose -f docker-compose.observability-server.yml restart grafana-training
    fi
    
    # Check if Loki config changed
    if git diff HEAD~1 HEAD --name-only 2>/dev/null | grep -q "loki-training-config.yaml"; then
        echo "Loki config changed, restarting Loki..."
        docker-compose -f docker-compose.observability-server.yml restart loki-training
    fi
fi
