import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io

# Load your model
MODEL_PATH = r"D:\Flutter\Main\backend\backend\Final-Group-Project-Backend\Cassava_Disease_Classification2.h5"
  # Update this to your actual model path
model = load_model(MODEL_PATH)
class_names = class_names = ['Cassava Bacterial Blight(CBB)',
 'Cassava Brown Streak Disease(CBSD)',
 'Cassava Green Mottle(CGM)',
 'Cassava Mosaic Disease(CMD)',
 'Healthy',
 'Others_Test']


def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((224, 224))
    image_array = img_to_array(image) / 255.0  # Normalize if the model expects normalized images
    return np.expand_dims(image_array, axis=0)

def predict_image(image_data):
    # Preprocess image
    img_array = preprocess_image(image_data)
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return predicted_class, confidence
