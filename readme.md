## Environment Setup

**Create a Conda environment for PyTorch**:
- **Create the environment**: `conda create --name pytorch`
- **Activate the environment**: `conda activate pytorch`

**Install required Python packages**:
- **Install dependencies**: `pip install -r requirements.txt`

## Audio Processing

Place your data in the following folder structure:
- **Audio/**


**To train the audio model**:
- **Change directory to Audio**: `cd Audio`
- **Run the training script**: `python audio_csv_train.py`

After training is complete, to evaluate the model and generate the confusion matrix:
- **Run the testing script**: `python audio_test.py`

## Image Processing

Place your image data in the following folder:
- **Image/**

**To train the image model using VGG16 as the base model**:
- **From the main folder, change directory to Image**: `cd Image`
- **Run the image training script**: `python image_train.py`

After training is complete, check the **models** folder in the parent directory for the last epoch's best model. Update the model path in `image_test.py`.

**To run testing and generate the confusion matrix**:
- **Run the testing script**: `python image_test.py`

## Running the UI

To start the Streamlit UI, be in the parent folder and execute:
- **Run the Streamlit app**: `streamlit run app.py`

The app will be available at **localhost:8501** in your web browser.
