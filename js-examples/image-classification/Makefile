.PHONY: fetch-model
fetch-model:
	curl -L https://github.com/onnx/models/raw/main/Computer_Vision/mobilenetv2_110d_Opset18_timm/mobilenetv2_110d_Opset18.onnx -o mobilenet.onnx

.PHONY: test-node
test-node:
	npm ci > /dev/null
	node classify-node.js ../../tools/test-images/sofa-cats.jpg
