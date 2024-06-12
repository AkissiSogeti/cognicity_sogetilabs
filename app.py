import gradio as gr
import pytesseract
import re
from hezar.models import Model

# Pour avoir la liste des langues disponibles
# print(pytesseract.get_languages(config=''))

# Get bounding box estimates
# print(pytesseract.image_to_boxes(Image.open('Images/voiture.jpg')))

# Get verbose data including boxes, confidences, line and page numbers
# print(pytesseract.image_to_data(Image.open('Images/voiture.jpg')))

# Get information about orientation and script detection
# print(pytesseract.image_to_osd(Image.open('Images/voiture.jpg')))

# Déclaration du css pour modifier le visuel
css = """
#input {background-color: #FFFAF0}
.feedback textarea {font-size: 24px !important}
.gradio-container {background-color: #FFDFB0}
"""

persianPlateModel = Model.load("hezarai/crnn-fa-64x256-license-plate-recognition")

def launch(image, radio):
    text = plate_to_text(image)

    if not text:
        return 'Nothing found'
    # Si moins de 7 caractères ou aucun chiffre = pas possible de trouver le numéro de plaque
    elif len(text) < 7 or not any(isinstance(item, str) and item.isdigit() for item in text):
        return 'Uncomplete plate'
    
    elif (radio == "French plate"):
        return french_plate(text)

    elif (radio == "Spain plate"):
        return spain_plate(text)
    
    elif (radio == "Italian plate"):
        return italian_plate(text)
      
    elif (radio == "Persian plate"):
        return(persianPlateModel.predict(image)[0]["text"])

def plate_to_text(image):
    # Récupération du texte issu de l'image
    plate = pytesseract.image_to_string(image)
    # Stockage des caractères (alphanumérique) du texte récupéré dans une liste pour pouvoir l'analyser et le retranscrire
    res = [i for i in list(plate) if (i.isdigit() or (i.isalpha() and i.isupper()))]
    return res

def french_plate(text):
    # On part du principe que les caractères représentant le numéro de plaque sont à la suite
    # Rappel du format : AA-000-AA
    while not text[2].isdigit():
            text.pop(0)
    while not len(text) == 7:
        if len(text) < 7:
            return 'Not french plate'
        else:
            text.pop(- 1)
    rest = ["".join(text[0:2]), "".join(text[2:5]), "".join(text[5:])]
    licence_plate_number = "-".join(rest)
    if verify_plate("France", licence_plate_number):
        return licence_plate_number
    else:
        return 'Not french plate'

def spain_plate(text):
    # On part du principe que les caractères représentant le numéro de plaque sont à la suite
    # Rappel du format : 0000 AAA
    while not text[0].isdigit():
            text.pop(0)
    while not len(text) == 7:
        if len(text) < 7:
            return 'Not spain plate'
        else:
            text.pop(- 1)
    rest = ["".join(text[0:4]), "".join(text[4:])]
    licence_plate_number = " ".join(rest)
    if verify_plate("Spain", licence_plate_number):
        return licence_plate_number
    else:
        return 'Not spain plate'

def italian_plate(text):
    # On part du principe que les caractères représentant le numéro de plaque sont à la suite
    # Rappel du format : AA 000AA
    while not text[2].isdigit():
            text.pop(0)
    while not len(text) == 7:
        if len(text) < 7:
            return 'Not italian plate'
        else:
            text.pop(- 1)
    rest = ["".join(text[0:2]), "".join(text[2:])]
    licence_plate_number = " ".join(rest)
    if verify_plate("Italia", licence_plate_number):
        return licence_plate_number
    else:
        return 'Not italian plate'

def verify_plate(country, plate):
    if (country == "France"):
        pattern = r'^([A-Z]{2}-\d{3}-[A-Z]{2})'
    elif (country == "Spain"):
        pattern = r'^(\d{4} [A-Z]{3})'
    elif (country == "Italia"):
        pattern = r'^([A-Z]{2} \d{3}[A-Z]{2})'
    return re.match(pattern, plate) is not None

# Création de l'interface gradio
with gr.Blocks(css=css) as demo:
    # Texte affiché en haut de la page
    gr.Markdown(
        """
        # Cognicity
        Put your licence plate image down below then launch the decision
        """)
    
    # Premier bloc
    with gr.Row():
        # Première colonne du premier bloc
        with gr.Column():
            image = gr.Image(label = "Input Image", type="pil",elem_classes = "component")
            radio = gr.Radio(["French plate", "Spain plate", "Italian plate", "Persian plate"],elem_classes = "component")
            # TODO Add buttons depending on the model used

        with gr.Column():
            launch_btn = gr.Button("Launch Decision")
            decision = gr.Textbox(label="Decision", value="Use the button to launch decision", elem_classes = "component")
 
            launch_btn.click(launch, [image, radio], decision)
        
demo.launch(share=True)