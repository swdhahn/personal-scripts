
docker_uuid=$(docker container ls | grep open-webui | awk '{print $1}')

echo "Found docker open-webui container with $docker_uuid"

docker stop $docker_uuid
docker rm $docker_uuid

docker run -d -e PORT=3000 -e WEBUI_AUTH=false -e OLLAMA_BASE_URL=http://127.0.0.1:11434 --network=host -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main



