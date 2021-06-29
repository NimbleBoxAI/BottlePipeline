#Imports
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
  """
      Returns the MobileNet Model
      Params
      ======
      path   (String)
          : path of the saved model
      device (String)
          : Device on which the model is to be loaded (GPU/CPU)
      Returns
      =======
      model (torchvision.models.mobilenetv3.MobileNetV3)
          : Model for Mobilenetv3
  """
  image_size = (256, 256)
  model = torch.load(path, map_location = device)
  model = model.to(device)
  return model

def get_transforms(image_size):
  """
      Returns a dictionary containing the transforms for Training and Validation Images. 
      Params
      ======
      image_size   (Tuple)
          :  Size of the images to be passed to the model
      Returns
      =======
      tfms (dict)
          : Transforms to be applied to the Training and Validation Images
  """
  tfms = transforms.Compose([
                  transforms.Resize(image_size),
                  transforms.ToTensor(),
                  transforms.Normalize(
                          [0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])
                          ])
  return tfms


def show_result(ref_img, img_list, tfms, device, tolerance):
  """
      Displays the results for the images uploaded
      Params
      ======
      ref_img  ( _io.BytesIO)
          : BytesIO Buffer of the  Reference Image
      img_list  (list)
          : List containing the BytesIO Buffer of Images uploaded for Inference
         
      tfms  (dict)
          : Transforms to be applied to the Training and Validation Images
      device  (String)
          : Device on which the model is to be loaded (GPU/CPU)
      tolerance (float)
          : Defines the tolerance value for the Color Detection Model    
  """  

  labels = ['Cap Present', 'Cap Missing']

  #Get the SegmentNet Model for BottleSegmentation 
  segment_net = get_model()
  #Get the MobileNetv3 Model for CapDetection
  model = get_mobilenet(path = "models/mobilenet-v3-small-best.pth", device = device)
  model.eval()

  #Pass the Reference Image to SegmentNet to remove noise
  if ref_img:
    ref = Image.open(ref_img).convert('RGB')
    img_t = transform_img(ref).unsqueeze(0)
    predictions = predict([ref], img_t, segment_net)[0]
    ref = Image.fromarray(predictions[0])
  
  if len(img_list) != 0:
    res = 0
    bar = st.progress(0)
    #Iterate over the images meant for Inference
    for prog, st_img in enumerate(img_list):
      st.write("\n", "-"*25,"\n")
      img = Image.open(st_img).convert('RGB')
      st.image(np.array(img), width = 200)
      img_t = transform_img(img).unsqueeze(0)
      predictions = predict([img], img_t, segment_net)[0]
      segmented_images = []
      image_labels = [[],[]]
      #Iterate over all the segmented images for a single inference image
      for j in predictions:
        img = Image.fromarray(j)
        img_t = tfms(img)
        img_t = torch.unsqueeze(img_t, 0).to(device)
        res = model(img_t)
        bar.progress(int(prog * 100/len(img_list)) + int(100/len(img_list)))
        segmented_images.append(img)
        image_labels[0].append(labels[torch.argmax(res)])

        if chromatic_match(ref, img, tolerance=tolerance):
          image_labels[1].append(True)
        else:
          image_labels[1].append(False)
          
      #Display the Result in a formatting of 3 columns
      j = 0
      while j < len(segmented_images):
        columns = st.beta_columns(3)
        for col in columns:
            col.write(" ")
        columns = st.beta_columns(3)
        for i in range(3):
            if j == len(segmented_images):
              break
            label = image_labels[0][j]
            if image_labels[1][j]:
              label = label + "\nColor matches with Reference"
            else:
              label = label + "\nColor Does not match with the Reference"
            columns[i].header(label)
            columns[i].image(segmented_images[j])
            j = j + 1

  else:
    st.text("Please Upload an image")


def main():
  """
    Driver Code
  """
  st.title("Bottle Processing Pipeline")
  st.header('Step 1: Upload reference image to set the desired color')
  st.write('Color of input bottle images are compared against the reference Image')

  #Get the Reference Image
  ref_img = st.file_uploader("Upload Reference Image", accept_multiple_files=False)
  if ref_img:
    ref = Image.open(ref_img).convert('RGB')
    st.image(ref, width=200)

  st.header('Step 2: Set Tolerance')
  tolerance_definition = "Tolerance sets the deviation in color tone from the reference image. Needs to be tuned."
  st.write(tolerance_definition)
  #Get the tolerance Value
  tolerance = st.number_input(label='Tolerance', value=0.2)

  st.header('Step 3: Upload Images.')
  st.write("""Upload picture(s) of Bottle(s) for prediction""")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  #Get All Images Inference is to be perfomed on
  img_list = st.file_uploader("Upload files here", accept_multiple_files=True)

  #Get Transforms, Infer and Display Results
  tfms = get_transforms(image_size=(256,256))
  show_result(ref_img, img_list, tfms, device, tolerance)

#Start the execution of the file from main()
if __name__ == '__main__':
	main()
