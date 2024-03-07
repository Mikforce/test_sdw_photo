# import requests
# import torch
# from PIL import Image
# from io import BytesIO
#
# from diffusers import CycleDiffusionPipeline, DDIMScheduler
#
#
# # load the scheduler. CycleDiffusion only supports stochastic schedulers.
#
# # load the pipeline
# # make sure you're logged in with `huggingface-cli login`
# model_id_or_path = "CompVis/stable-diffusion-v1-4"
# scheduler = DDIMScheduler.from_pretrained(model_id_or_path, subfolder="scheduler")
# pipe = CycleDiffusionPipeline.from_pretrained(model_id_or_path, scheduler=scheduler).to("cpu")
#
# # let's download an initial image
# url = "https://raw.githubusercontent.com/ChenWu98/cycle-diffusion/main/data/dalle2/An%20astronaut%20riding%20a%20horse.png"
# response = requests.get(url)
# print(response.content)
# init_image = Image.open(BytesIO(response.content)).convert("RGB")
# init_image = init_image.resize((512, 512))
# init_image.save("horse.png")
#
# # let's specify a prompt
# source_prompt = "An astronaut riding a horse"
# prompt = "An astronaut riding an elephant"
#
# # call the pipeline
# image = pipe(
#     prompt=prompt,
#     source_prompt=source_prompt,
#     image=init_image,
#     num_inference_steps=100,
#     eta=0.1,
#     strength=0.8,
#     guidance_scale=2,
#     source_guidance_scale=1,
# ).images[0]
#
# image.save("horse_to_elephant.png")










import requests
from flask import Flask, render_template, request, send_file
from PIL import Image
from io import BytesIO
from diffusers import CycleDiffusionPipeline, DDIMScheduler

app = Flask(__name__)

# загружаем планировщик. CycleDiffusion поддерживает только стохастические планировщики.
# загружаем конвейер
model_id_or_path = "CompVis/stable-diffusion-v1-4"
scheduler = DDIMScheduler.from_pretrained(model_id_or_path, subfolder="scheduler")
pipe = CycleDiffusionPipeline.from_pretrained(model_id_or_path, scheduler=scheduler).to("cuda") # если нет cuda изменить на cpu

@app.route('/')
def upload_file():
    return render_template('index.html')
@app.route('/', methods=['POST'])
def upload_image():
    # давайте загрузим исходное изображение
    prompt = request.form.get('prompt', '')
    source_prompt = request.form.get('source_prompt', '')
    file = request.files['file']
    init_image = Image.open(file).convert("RGB")
    init_image = init_image.resize((512, 512))
    # давайте укажем подсказку
    num_inference_steps = int(request.form.get('num_inference_steps', ''))
    strength_str = request.form.get('strength', '').replace(',', '.')
    strength = float(strength_str) if strength_str else 0.0

    eta_str = request.form.get('strength', '').replace(',', '.')
    eta = float(eta_str) if strength_str else 0.0
    guidance_scale = int(request.form.get('guidance_scale', ''))
    source_guidance_scale = int(request.form.get('source_guidance_scale', ''))

    # вызов конвейера
    image = pipe(
        prompt=prompt,
        source_prompt=source_prompt,
        image=init_image,
        num_inference_steps=num_inference_steps,
        eta=eta,
        strength=strength,
        guidance_scale=guidance_scale,
        source_guidance_scale=source_guidance_scale,
    ).images[0]

    image.save("horse_to_elephant.png")

    return send_file(image, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)