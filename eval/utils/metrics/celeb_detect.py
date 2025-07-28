import os

from model_training.utils import preprocess_image
from model_training.helpers.labels import Labels
from model_training.helpers.face_recognizer import FaceRecognizer
from model_training.utils import evenly_spaced_sampling
from model_training.preprocessors.face_detection.face_detector import FaceDetector

from skimage import io

def celeb_detect(image_path):
    model_labels = Labels(resources_path="/home/nkusumba/T2I_Eval_Benchmark/utils/metrics/resources")
    face_detector = FaceDetector(
        "/home/nkusumba/T2I_Eval_Benchmark/utils/metrics/resources",
        margin=float(0.2),
        use_cuda="false" == "true"
    )
    face_recognizer = FaceRecognizer(
        labels=model_labels,
        resources_path="/home/nkusumba/T2I_Eval_Benchmark/utils/metrics/resources",
        use_cuda="false" == "true",
        top_n=5
    )

    image_size = 224

    def process_image(path):
        image = io.imread(path)
        face_images = face_detector.perform_single(image)
        face_images = [preprocess_image(image, image_size) for image, _ in face_images]
        return face_recognizer.perform(face_images)

    predictions  = []
    for path in image_path:
        predictions.append(process_image(path))

    return predictions