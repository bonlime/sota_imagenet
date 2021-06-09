docker build -f docker/Dockerfile --build-arg RANDOM_VAR=$(date +%s) -t bonlime/imagenet:latest .
