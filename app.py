import os
import cv2
import numpy as np
import gradio as gr
from RPV import main,add_missing_data,visualize

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown(
            '''# Gradio
            Use one of the models to analyse the image or the video
            '''
        )
        with gr.Row():
            # select model
            model_type = gr.Dropdown(["RPV","DOVA","XAI"], label="Select Model")

    # SAM parameters
    with gr.Accordion(label='Parameters', open=False):
        with gr.Row():
            points_per_side = gr.Number()

    # Segment image
    with gr.Tab(label='Image Input'):
        with gr.Row():
            with gr.Column():
                # input image
                input_image = gr.Image(type="numpy")
              # run button
                button_image = gr.Button("Launch model!")
            with gr.Tab(label='Image Output'):
                output_image = gr.Image(type='numpy')


    # Segment video
    with gr.Tab(label='Video Input'):
        with gr.Row():
            with gr.Column():
                input_video = gr.Video()
                with gr.Row():
                    button_video = gr.Button("Launch model!")
            with gr.Tab(label='Video Output'):
                output_video = gr.Video(format='mp4')

    def create_image(input, model):
        return ("")

    def create_video(input, model):
        if model == "RPV":
            print("Lancement du modèle")
            main.lanch_model(input)
            print("Interpolation")
            add_missing_data.add_interpolation()
            print("Création de la vidéo")
            visualize.create_final_video(input)
            print("Fini !")
            return("RPV/out.mp4")

    # button image
    button_image.click(create_image,
                    inputs=[input_image, model_type],
                    outputs=[output_image])
    # button video
    button_video.click(create_video,
                       inputs=[input_video, model_type],
                       outputs=[output_video])

demo.queue().launch()
