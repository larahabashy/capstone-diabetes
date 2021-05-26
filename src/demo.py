import streamlit as st
import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms, models, datasets, utils
from PIL import Image
import altair as alt

st.title('Lipohypertrophy Prediction')

#load the model
cnn_model = models.inception_v3(pretrained=True)
cnn_model.fc = nn.Linear(2048, 1)
cnn_model.aux_logits = False

cnn_model.load_state_dict(torch.load('test_auto/models/cnn_model.pt')) #put the directory here where cnn_model.pt is located
cnn_model.eval()
left_column, right_column = st.beta_columns(2)

#make prediction
uploaded_file = left_column.file_uploader('Upload ultrasound image here!')

if uploaded_file is not None:
    uploaded_file = Image.open(uploaded_file)
    st.image(uploaded_file.resize((200, 200)), caption='Uploaded Image', use_column_width=False)
    image_tensor = transforms.functional.to_tensor(uploaded_file.resize((300, 300))).unsqueeze(0)

    pos = float(torch.sigmoid(cnn_model(image_tensor)).detach().numpy())
    neg = 1 - pos
    pred = 'Positive' if pos >0.5 else 'Negative' 
else:
    pass

#display results
with right_column:
    if uploaded_file is not None:
        st.write(f"Lipohypertrophy prediction: {pred}")
        chart_data = pd.DataFrame(
            {'label': ['Positive','Negative'],
            'value': [pos,neg],
            'perc':[str(round(pos*100,1))+'%', str(round(neg*100,1))+'%']})
        chart = alt.Chart(chart_data,title="Prediction Confidence").mark_bar().encode(
        alt.X("value", title = "", axis = None),
        alt.Y("label", title = "", sort=['Positive','Negative']),
        alt.Text('value'),
        alt.Color('label', legend = None)
        )
        text = chart.mark_text(
        align='left',
        baseline='middle',
        dx=3 
        ).encode(
            text='perc'
        )
        st.write(alt.layer(chart, text).configure_axis(
        grid=False).configure_view(
        strokeWidth=0
        ))
    else:
        st.write()
        