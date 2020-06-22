echo "creating waggle network"
docker network create waggle &> /dev/null

echo "running rabbitmq"
docker run -d --restart always --network waggle \
    -p 5672:5672 \
    --name rabbitmq \
    waggle/rabbitmq:ep
