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

    image.save("rez.png")

    return send_file(image, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)