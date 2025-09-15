#!/bin/bash
# revert-permissions.sh

# Obtener el usuario actual
CURRENT_USER=$(whoami)
CURRENT_UID=$(id -u)
CURRENT_GID=$(id -g)

# Revertir ownership al usuario actual
sudo chown -R $CURRENT_UID:$CURRENT_GID ./data ./model ./src

echo "Permisos revertidos al usuario: $CURRENT_USER ($CURRENT_UID:$CURRENT_GID)"