import paddle
model_path = r"d:\Computer Vision Project\models\inference\rec_vi_plate\inference"
# load model
model = paddle.jit.load(model_path)
print("Model Inputs:", model.inputs)
print("Model Outputs:", model.outputs)
