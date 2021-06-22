from os import path
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from infer import get_model, predict, transform_img
from color import chromatic_match


def get_mobilenet(path, device):
  image_size = (256, 256)
  model = torch.load(path, map_location = device)
  model = model.to(device)
  return model

def get_transforms(image_size):
  tfms = transforms.Compose([
                  transforms.Resize(image_size),
                  transforms.ToTensor(),
                  transforms.Normalize(
                          [0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])
                          ])
  return tfms


def show_result(ref_img, img_list, tfms, device, tolerance):
  labels = ['Cap Present', 'Cap Missing']
  segment_net = get_model()
  model = get_mobilenet(path = "models/mobilenet-v3-small-best.pth", device = device)
  model.eval()

  if ref_img:
    ref = Image.open(ref_img).convert('RGB')
    img_t = transform_img(ref).unsqueeze(0)
    predictions = predict([ref], img_t, segment_net)[0]
    ref = Image.fromarray(predictions[0])

  

  if len(img_list) != 0:
    res = 0
    bar = st.progress(0)

    for prog, st_img in enumerate(img_list):

      img = Image.open(st_img).convert('RGB')
      st.image(np.array(img), width = 200)
      img_t = transform_img(img).unsqueeze(0)
      predictions = predict([img], img_t, segment_net)[0]

      for j in predictions:

        img = Image.fromarray(j)
        img_t = tfms(img)
        img_t = torch.unsqueeze(img_t, 0).to(device)

        res = model(img_t)
        bar.progress(int(prog * 100/len(img_list)) + int(100/len(img_list)))
        st.image(img, width=100)
        st.text("Label: " + labels[torch.argmax(res)])

        if chromatic_match(ref, img, tolerance=tolerance):
          st.write('Image color matches with reference.')
        else:
          st.write('Image color does not match with reference.')
  else:
    st.text("Please Upload an image")


def main():
  st.title("Bottle Processing Pipeline")
  
  st.header('Step 1: Upload Reference Image')
  ref_img = st.file_uploader("Upload Reference Image", accept_multiple_files=False)
  if ref_img:
    ref = Image.open(ref_img).convert('RGB')
    st.image(ref, width=200)

  st.header('Step 2: Set Tolerance')
  tolerance = st.number_input(label='Tolerance')

  st.header('Step 3: Upload Images.')
  st.write("""Upload pictures of Bottle for prediction, you can also upload multiple
            pictures of Bottles to predict multiple results or combine them to improve
            results.""")
  st.write("Use the below checkbox for that selection before uploading images")
  combine = st.checkbox("Combine images for the result")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  img_list = st.file_uploader("Upload files here", accept_multiple_files=True)
  tfms = get_transforms(image_size=(256,256))
  show_result(ref_img, img_list, tfms, device, tolerance)

  
if __name__ == '__main__':
	main()