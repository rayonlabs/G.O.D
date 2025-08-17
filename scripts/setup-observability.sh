#!/bin/bash

set -e

echo "=== Training Observability Setup Script ==="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo -e "${RED}Please do not run this script as root${NC}"
   exit 1
fi

# Function to create directories
create_directories() {
    echo "Creating necessary directories..."
    mkdir -p config/nginx/ssl
    mkdir -p config/vector
    echo -e "${GREEN}✓ Directories created${NC}"
}

# Function to generate self-signed SSL certificates
generate_ssl_certificates() {
    echo
    echo "Generating SSL certificates..."
    
    if [ -f "config/nginx/ssl/cert.pem" ] && [ -f "config/nginx/ssl/key.pem" ]; then
        echo -e "${YELLOW}SSL certificates already exist. Skipping...${NC}"
        return
    fi
    
    # Get domain name
    read -p "Enter your domain name (or IP address) for the observability server: " DOMAIN
    
    # Generate private key and certificate
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout config/nginx/ssl/key.pem \
        -out config/nginx/ssl/cert.pem \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=$DOMAIN"
    
    chmod 400 config/nginx/ssl/key.pem
    chmod 444 config/nginx/ssl/cert.pem
    
    echo -e "${GREEN}✓ SSL certificates generated${NC}"
}

# Function to create htpasswd file for basic auth
create_htpasswd() {
    echo
    echo "Setting up authentication..."
    
    # Check if htpasswd is installed
    if ! command -v htpasswd &> /dev/null; then
        echo -e "${YELLOW}htpasswd not found. Installing apache2-utils...${NC}"
        sudo apt-get update && sudo apt-get install -y apache2-utils
    fi
    
    # Create users
    echo "Creating users for Loki access..."
    
    # Admin user
    read -p "Enter admin username (default: admin): " ADMIN_USER
    ADMIN_USER=${ADMIN_USER:-admin}
    htpasswd -c config/nginx/.htpasswd $ADMIN_USER
    
    # Trainer user (for log shipping)
    read -p "Enter trainer username for log shipping (default: trainer): " TRAINER_USER
    TRAINER_USER=${TRAINER_USER:-trainer}
    htpasswd config/nginx/.htpasswd $TRAINER_USER
    
    chmod 644 config/nginx/.htpasswd
    
    echo -e "${GREEN}✓ Authentication configured${NC}"
}

# Function to create environment file
create_env_file() {
    echo
    echo "Creating environment configuration..."
    
    if [ -f ".env.observability" ]; then
        echo -e "${YELLOW}.env.observability already exists. Backing up...${NC}"
        cp .env.observability .env.observability.backup.$(date +%Y%m%d_%H%M%S)
    fi
    
    read -p "Enter observability server domain/IP: " OBS_DOMAIN
    read -p "Enter Grafana admin password: " -s GRAFANA_PASS
    echo
    read -p "Enter Loki username for trainers: " LOKI_USER
    read -p "Enter Loki password for trainers: " -s LOKI_PASS
    echo
    read -p "Enable anonymous Grafana viewing? (y/n): " ANON_VIEW
    
    if [ "$ANON_VIEW" = "y" ]; then
        ANON_ENABLED="true"
    else
        ANON_ENABLED="false"
    fi
    
    cat > .env.observability << EOF
# Observability Server Configuration
OBSERVABILITY_DOMAIN=$OBS_DOMAIN

# Grafana Configuration
GRAFANA_TRAINING_USERNAME=admin
GRAFANA_TRAINING_PASSWORD=$GRAFANA_PASS
GRAFANA_ANONYMOUS_ENABLED=$ANON_ENABLED

# Loki Configuration  
LOKI_AUTH_ENABLED=false
LOKI_ENDPOINT=https://$OBS_DOMAIN:3101
LOKI_USERNAME=$LOKI_USER
LOKI_PASSWORD=$LOKI_PASS

# Vector Configuration
VECTOR_LOG_LEVEL=info
ENVIRONMENT=production

# Prometheus Targets (comma-separated)
# Example: trainer1.example.com:9090,trainer2.example.com:9090
VECTOR_TARGETS=
NODE_EXPORTER_TARGETS=
EOF
    
    chmod 600 .env.observability
    
    echo -e "${GREEN}✓ Environment file created${NC}"
}

# Function to create trainer environment file
create_trainer_env() {
    echo
    echo "Creating trainer server environment file..."
    
    cat > .env.trainer << EOF
# Trainer Server Configuration
# Copy this file to each trainer server

# Loki endpoint for log shipping
LOKI_ENDPOINT=https://$OBS_DOMAIN:3101
LOKI_USERNAME=$LOKI_USER
LOKI_PASSWORD=$LOKI_PASS

# Vector settings
VECTOR_LOG_LEVEL=info
ENVIRONMENT=production
EOF
    
    chmod 600 .env.trainer
    
    echo -e "${GREEN}✓ Trainer environment file created${NC}"
}

# Function to show deployment instructions
show_instructions() {
    echo
    echo "========================================="
    echo -e "${GREEN}Setup Complete!${NC}"
    echo "========================================="
    echo
    echo "Next steps:"
    echo
    echo "1. On the OBSERVABILITY SERVER:"
    echo "   - Copy .env.observability to the server"
    echo "   - Run: docker-compose -f docker-compose.observability-server.yml --env-file .env.observability up -d"
    echo "   - Access Grafana at: https://$OBS_DOMAIN:3001"
    echo
    echo "2. On each TRAINER SERVER:"
    echo "   - Copy .env.trainer to the server"  
    echo "   - Copy config/vector/vector.toml to the server"
    echo "   - Run: docker-compose -f docker-compose.trainer-server.yml --env-file .env.trainer up -d"
    echo
    echo "3. Update trainer code to use network mode 'bridge' instead of 'none'"
    echo
    echo "4. Test log shipping:"
    echo "   - Run a training task"
    echo "   - Check Grafana for logs"
    echo
    echo -e "${YELLOW}Important Security Notes:${NC}"
    echo "- Replace self-signed certificates with proper SSL certificates in production"
    echo "- Use strong passwords for all accounts"
    echo "- Restrict network access using firewall rules"
    echo "- Regularly update all container images"
    echo
}

# Main execution
main() {
    echo "This script will set up the training observability stack"
    echo "for multi-server deployment."
    echo
    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    
    create_directories
    generate_ssl_certificates
    create_htpasswd
    create_env_file
    create_trainer_env
    show_instructions
}

# Run main function
main