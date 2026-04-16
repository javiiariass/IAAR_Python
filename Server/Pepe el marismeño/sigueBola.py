import cv2
import numpy as np
import time
import onnxruntime as ort
from motor import tankMotor 

class sigueBola:

    idBolaRoja = 0  # ID de la clase de la bola roja, actualizar según el modelo
    model_path = "path/to/your/model.onnx"  # Update with your model path
               

    def __init__(self):
        model = self.load_model(self.model_path)

        self.PWM = tankMotor()

        picam2 = Picamera2()
        picam2.start()  
     
    def analizar_orientacion_bola(self, boxes, scores, class_ids, frame_width, frame_height, umbral_confianza=0.5):
        for box, score, class_id in zip(boxes[0], scores[0], class_ids[0]):
            if int(class_id) == self.idBolaRoja and score > umbral_confianza:  # BOLA_ROJA
                x1, y1, x2, y2 = box.astype(int)
                centro_x = (x1 + x2) // 2
                centro_y = (y1 + y2) // 2

                margen_x = frame_width * 0.1
                margen_y = frame_height * 0.1

                # Dirección horizontal
                if centro_x < (frame_width / 2 - margen_x):
                    dir_h = "IZQUIERDA"
                elif centro_x > (frame_width / 2 + margen_x):
                    dir_h = "DERECHA"
                else:
                    dir_h = "CENTRO"

                # Dirección vertical
                if centro_y < (frame_height / 2 - margen_y):
                    dir_v = "ARRIBA"
                elif centro_y > (frame_height / 2 + margen_y):
                    dir_v = "ABAJO"
                else:
                    dir_v = "CENTRO"

                return dir_h, dir_v, (x1, y1, x2, y2)
        return None, None, None

    def load_model(self,model_path):
        """
        Load the ONNX model from the specified path.
        """
        try:
            session = ort.InferenceSession(model_path)
            return session
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
        

    def preprocess_image(self, image):
        """
        Preprocess the input image for the model.
        Resize to 224x224 and normalize pixel values.
        """
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # Change to CHW format
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image

    def predict(self, model, image):
        """
        Predict the class of the input image using the model.
        """
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        input_data = self.preprocess_image(image)
        
        try:
            result = model.run([output_name], {input_name: input_data})
            boxes, scores, class_ids = result[0], result[1], result[2]
            return boxes, scores, class_ids
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

    def getImage(self):
        frame_rgb = self.picam2.capture_array()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def moverMotor(self, orientacion):

        '''
        SI VA MUY RAPIDO CAMBIAR LA VELOCIDAD DE LOS MOTORES DE 2000 A 1000 O MENOS
        '''

        if orientacion[0] == "IZQUIERDA":
            self.PWM.setMotorModel(0, 2000)  
            print("Mover motor a la izquierda")

        elif orientacion[0] == "DERECHA":
            self.PWM.setMotorModel(2000, 0)
            print("Mover motor a la derecha")
    
        elif orientacion[0] == "CENTRO":
            self.PWM.setMotorModel(2000, 2000)  
            print("Mantener motor en el centro")
        
        

    def sigueBolas(self):
        
        # Load an example image
        image = self.getImage()  # Replace with your image loading function or path

        if image is None:
            print("Image could not be loaded. Exiting.")
            return
        
        # Predict the class of the image
        boxes, scores, class_ids = self.predict(self.model, image)
        
        if boxes is not None:
            print("------------------------------")
            orientacion = self.analizar_orientacion_bola(boxes, scores, class_ids, image.shape[1], image.shape[0])
            print(f"Orientación de la bola: {orientacion[0]}, {orientacion[1]}")
            print(f"Coordenadas de la bola: {orientacion[2]}")
            print("------------------------------")
            print("\n\n")

            self.moverMotor(orientacion, self.PWM)
            time.sleep(0.1)  # Esperar un poco antes de la siguiente iteración

        else:
            print("Prediction failed.")