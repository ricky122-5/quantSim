IMAGE_NAME = quant-simulator
CONTAINER_NAME = quant-sim-container

build:
	docker build -t $(IMAGE_NAME) .

run: build
	docker run --name $(CONTAINER_NAME) -p 8080:80 $(IMAGE_NAME)

stop:
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)

clean:
	docker rmi $(IMAGE_NAME)

rebuild: clean build run
