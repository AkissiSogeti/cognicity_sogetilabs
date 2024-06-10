import gradio as gr
from hezar.models import Model
from tesserocr import PSM, image_to_text

css = """
.component {background-color: #A2D2FF}
.feedback textarea {font-size: 24px !important}
.gradio-container {background-color: #FFDFB0}
"""

persianPlateModel = Model.load("hezarai/crnn-fa-64x256-license-plate-recognition")

def launch(image, radio):

    if (radio == "Persian plate"):
        return(persianPlateModel.predict(image)[0]["text"])

    # TODO Add other processing depending on the model used

with gr.Blocks(css=css) as demo:

    gr.Markdown(
        """
        # Cognicity
        Put your image down below then launch the decision !
        """)

    with gr.Row():
        with gr.Column():
            image = gr.Image(label = "Input Image", type="pil",elem_classes = "component")
            radio = gr.Radio(["Persian plate"],elem_classes = "component")
            # TODO Add buttons depending on the model used

        with gr.Column():
            launch_btn = gr.Button("Launch Decision")
            decision = gr.Textbox(label="Decision", value="Use the button to launch decision", elem_classes = "component")
 
            launch_btn.click(launch, [image, radio], decision)

demo.launch()