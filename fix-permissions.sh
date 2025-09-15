#!/bin/bash
# fix-permissions.sh

# Crear directorios si no existen
mkdir -p ./data ./model ./src/config

# Cambiar ownership a Airflow user (UID 50000)
sudo chown -R 50000:0 ./data ./model
sudo chmod -R 755 ./data ./model

# Asegurar que src también tenga permisos correctos
sudo chown -R 50000:0 ./src
sudo chmod -R 755 ./src

echo "Permisos fijados correctamente"