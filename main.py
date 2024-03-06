from flask import Flask, render_template, request, send_file
from diffusers import KandinskyImg2ImgPipeline, KandinskyPriorPipeline
import torch
from PIL import Image
from io import BytesIO


app = Flask(__name__)

pipe_prior = KandinskyPriorPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float32
)  #  в случае ошибки изменить на 16
pipe_prior.to("cpu")  # если нет cuda изменить на cpu


@app.route('/')
def upload_file():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    prompt = request.form.get('prompt', '')
    num_inference_steps = int(request.form.get('num_inference_steps', ''))
    strength_str = request.form.get('strength', '').replace(',', '.')
    strength = float(strength_str) if strength_str else 0.0

    image_emb, zero_image_emb = pipe_prior(prompt, return_dict=False)
    file = request.files['file']
    image = Image.open(file).convert("RGB")

    pipe = KandinskyImg2ImgPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-1", torch_dtype=torch.float32
    )  # в случае ошибки изменить на 16
    pipe.to("cpu")  # если нет cuda изменить на cpu

    processed_images  = pipe(
        prompt,
        image=image,
        image_embeds=image_emb,
        negative_image_embeds=zero_image_emb,
        height=768,
        width=768,
        num_inference_steps=num_inference_steps,
        strength=strength,
    ).images

    img_io = BytesIO()
    processed_images[0].save(img_io, format='PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)