.PHONY: format build train                                                                        

format:
	black --line-length 120 .
build:
	docker build -f docker/Dockerfile --build-arg RANDOM_VAR=$(date +%s) -t bonlime/imagenet:latest .