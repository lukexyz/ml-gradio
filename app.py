import gradio as gr
from fastai.vision.all import *

learn = load_learner('export.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Pet Breed Classifier"
description = "A pet breed classifier trained on the Oxford Pets dataset with fastai."
article="<p style='text-align: center'><a href='https://tmabraham.github.io/blog/gradio_hf_spaces_tutorial' target='_blank'>Blog post</a></p>"

examples = ['dog.jpg', 'cat.jpg', 'elephant_shrew.jpg']

gr.Interface(fn=predict,
             inputs=gr.inputs.Image(shape=(512, 512)),
             outputs=gr.outputs.Label(num_top_classes=3),
             title=title,
             description=description,
             article=article,
             examples=examples,
             interpretation='default').launch(share=True), enable_queue=True
