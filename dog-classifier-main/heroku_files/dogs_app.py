import os
import time
import requests
from PIL import ExifTags
from fastai.learner import load_learner
from fastai.vision.core import PILImage
import streamlit as st
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

AWS_DIR = 'https://wjdogs.s3.amazonaws.com'
MODEL_FILE = 'dogs_online_resnet50_cpu.pkl'

st.set_page_config(
    page_title="Dog Classifier App",
    page_icon="🐶",
    
)


html_temp = """
<div style ="background-color:gray;"><p style="color:black; font_size:50px;"><center/>Dog Breed Classifier</p></div>
            """
st.markdown(html_temp,unsafe_allow_html=True)

st.image("https://previews.123rf.com/images/illustratiostock/illustratiostock1607/illustratiostock160700202/59304326-jeu-de-diff%C3%A9rentes-races-de-chiens-sur-un-fond-blanc.jpg")


st.write("This project classifies dog photos using a CNN fine-tuned from ResNet-50 in fastai.")


file_data = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])


def download_file(url):
    with st.spinner('Downloading model...'):
        
        local_filename = url.split('/')[-1]
        
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        return local_filename

def fix_rotation(file_data):
    try:
        image = PILImage.create(file_data)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break

        exif = dict(image.getexif().items())

        rot = 0
        if exif[orientation] == 3:
            rot = 180
        elif exif[orientation] == 6:
            rot = 270
        elif exif[orientation] == 8:
            rot = 90

        if rot != 0:
            st.write(f"Rotating image {rot} degrees (you're probably on iOS)...")
            image = image.rotate(rot, expand=True)
            image.__class__ = PILImage

    except (AttributeError, KeyError, IndexError):
        pass  # image didn't have EXIF data

    return image



@st.cache(allow_output_mutation=True)
def get_model():
    if not os.path.isfile(MODEL_FILE):
        _ = download_file(f'{AWS_DIR}/{MODEL_FILE}')

    learn = load_learner(MODEL_FILE)
    return learn

learn = get_model()

if file_data is not None:
    with st.spinner('Classifying...'):
        img = fix_rotation(file_data)
        
        st.write('## Your Dog Image')
        st.image(img, width=700)
        
        my_bar = st.progress(0)

        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)

        # classify
        pred, pred_idx, probs = learn.predict(img)
        top3_preds = sorted(list(zip(learn.dls.vocab, list(probs.numpy()))), key=lambda x: x[1], reverse=True)[:3]

        # prepare output
        out_text = '<table><tr> <th>Dog Breed</th> <th>Confidence</th> <th>Some Example</th> <th>Some information</th> </tr>'

        for pred in top3_preds:
            example = AWS_DIR + '/' + pred[0].replace(" ", "").lower() + ".jpg"
            out_text += '<tr>' + \
                            f'<td>{pred[0]}</td>' + \
                            f'<td>{100 * pred[1]:.02f}%</td>' + \
                            f'<td><img src="{example}" height="200" /></td>' + \
                            f'<td></td>' + \
                        '</tr>'
        out_text += '</table><br><br>'
        st.balloons()


        html_temp = """
        <div ><p style="color:black; font_size:30px;"><center/>What the model thinks</p></div>
                    """
        st.markdown(html_temp,unsafe_allow_html=True)
        
        st.markdown(out_text, unsafe_allow_html=True)

        st.write(f"🤔 List of dog breeds used in dog classifier, [click here]({AWS_DIR}/dog_breeds.html).")
        st.markdown('By <a href="https://www.linkedin.com/in/salah-sammari-60303419b/" target="_blank">salah sammari</a>' , unsafe_allow_html=True)