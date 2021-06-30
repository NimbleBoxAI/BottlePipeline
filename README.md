# BottlePipeline

The pipeline involves [Bottle Segmentation](https://github.com/NimbleBoxAI/BottleSegmentation), [Bottle Color Detection](https://github.com/NimbleBoxAI/BottleColourDetector) & [Bottle Cap Detection](https://github.com/NimbleBoxAI/CapDetection).

Files:
- 'infer.py': Bottle Segmentation and cap detection
- 'color.py': Color Detection
- 'app.py': streamlit webapp for this example

### Usage

Simply run the Streamlit app to use this tool ->
```

$ streamlit run app.py

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.4:8501

```
This command should automatically open up a browser with the network URL.
The App will prompt you to upload a few Images and set a few parameters (All details will be visible inside the app itself).
After all the steps have been executed, the app will then classify the images and display the results.

<img src="./usage.gif">
