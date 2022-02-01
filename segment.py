import numpy as np
import json
import base64
from io import BytesIO
# import boto3
import sys
import tempfile
import os
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import Keras.backend as K
from tensorflow.keras.losses import binary_crossentropy
from PIL import Image
import glob
import re 

def get_iou_vector(A, B):
    # Numpy version    
    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)
        
        # deal with empty mask first
        if true == 0:
            metric += (pred == 0)
            continue
        
        # non empty mask case.  Union is never empty 
        # hence it is safe to divide by its number of pixels
        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union
        
        # iou metrric is a stepwise approximation of the real iou over 0.5
        iou = np.floor(max(0, (iou - 0.45)*20)) / 10
        
        metric += iou
        
    # teake the average over all images in batch
    metric /= batch_size
    return metric


def my_iou_metric(label, pred):
    # Tensorflow version
    # return tf.numpy_function(get_iou_vector, [label, pred > 0.5], tf.float64)
    return get_iou_vector(label, pred > 0.5)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))

def img_to_base64_str(img):
    buffered=BytesIO()
    img.save(buffered,format="PNG")
    buffered.seek(0)
    img_bytes=buffered.getvalue()
    img_str="data:image/png;base64,"+base64.b64encode(img_bytes).decode()
    
    return img_str

def get_model():
    model=load_model('BaseU.hdf5',custom_objects={'my_iou_metric': my_iou_metric, 'dice_coef': dice_coef, 'dice_loss': dice_loss, 'bce_dice_loss': bce_dice_loss, 'bce_logdice_loss': bce_logdice_loss})
    return model

def predict_scan(filename):
    original=Image.open(filename)
    image =[cv2.resize(np.array(Image.open(filename)),(256,256))]
    # numpy_image=img_to_array(original)
    img=np.array(image)
    img=np.expand_dims(img,axis=-1)
    img=np.broadcast_to(img,(1,256,256,3)).copy()
    print("Making predictions")
    model=get_model()
    prediction=model.predict(img)
    
    val_pred=prediction.squeeze()
    pred_mask=np.array(np.round(val_pred>0.5),dtype=np.float32)
    pred_mask_image = (pred_mask*255).astype(np.uint8)
    print(pred_mask_image[pred_mask_image>2])
    pred_mask_image=Image.fromarray(pred_mask_image)
    

    
    pred_mask_image=pred_mask_image.convert("L")
    pred_mask_image.save('gfg_dummy_pic.png')
    # print(pred_mask[pred_mask>0.5])
    print("Done")
    
    
    return img_to_base64_str(pred_mask_image)
    
   
def predict_base64_image(name, contents):
    fd, file_path = tempfile.mkstemp()
    with open(fd,'wb') as f:
        f.write(base64.b64decode(contents))

    prediction = predict_scan(file_path)
    os.remove(file_path)
    return {name:prediction} 
    
if __name__ == "__main__":
    for filename in glob.glob(f"data/*.PNG"):
        classes = predict_scan(filename)
        format, imgstr = classes.split(';base64,')
        
        input_file = os.path.basename(filename)

        output_file_name = re.sub(".JPEG", "_pred.json", input_file)

        print(f"{output_file_name}: {imgstr}")

        # with open(f"output/{output_file_name}", "w") as f:
        #     # json.dump(dict(file=input_file, classes=classes), f)
        #     f.write(base64.decodebytes(imgstr))
