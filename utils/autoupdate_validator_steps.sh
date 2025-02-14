# Steps to take to update the validator automatically
# Change each time but take caution

deactivate || true
rm -rf $HOME/.venv
python -m venv $HOME/.venv
chown -R $SUDO_USER:$SUDO_USER $HOME/.venv $HOME/.bashrc
source $HOME/.venv/bin/activate
pip install -e .

task validator
