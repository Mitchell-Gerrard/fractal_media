import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm
class photo_fractal:
    def __init__(self, image_path):
        self.image_path = image_path
        self.fractal_image = None
        self.fractal_exist=False
        self.dpi=[100,100]
    def normalise_complex(self,Z):
        # Normalize real and imaginary parts independently to [-1,1]
        real = Z.real
        imag = Z.imag
        
        real_norm = 2 * (real - real.min()) / (real.max() - real.min()) - 1
        imag_norm = 2 * (imag - imag.min()) / (imag.max() - imag.min()) - 1
        
        return real_norm + 1j * imag_norm

    def generate_fractal(self, num_iterations=5, shift=0.1,scale=2):
        image=Image.open(self.image_path).convert("RGB")
        img = np.array(image)
        self.dpi=image.info['dpi']
        height, width = img.shape[:2]

        aspect_ratio = width / height
        x = np.linspace(-scale * aspect_ratio, scale * aspect_ratio, width)
        y = np.linspace(-scale, scale, height)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y

        # Apply complex function
        for _ in range(num_iterations):
            # Shift the complex plane
            Z = Z - shift

            # Apply a transformation (e.g., square root)
            Z = Z ** 2
        #Z = self.normalise_complex(Z) 
        Z_transformed = Z 


        X_new = ((Z_transformed.real + scale * aspect_ratio) / (2 * scale * aspect_ratio)) * (width - 1)
        Y_new = ((Z_transformed.imag + scale) / (2 * scale)) * (height - 1)
        # Convert to float32 for OpenCV
        map_x = X_new.astype(np.float32)
        map_y = Y_new.astype(np.float32)
        map_x = np.clip(map_x, 0, width - 1).astype(np.float32)
        map_y = np.clip(map_y, 0, height - 1).astype(np.float32)
        # Remap image
        remapped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

        # Show result
        plt.imshow(remapped)
        plt.axis('off')
        plt.show()
        self.fractal_image = remapped
        self.fractal_exist=True
    
        
        
    def save_fractal(self, output_path,dpi=300):
        if self.fractal_exist:
            plt.imshow(self.fractal_image)
            plt.axis('off')

            plt.savefig(output_path,bbox_inches='tight',pad_inches=0,dpi=dpi)
            print(f"Fractal saved to {output_path}")
        else:
            print("No fractal generated yet.")
class video_fractal:
    def __init__(self):
        
        self.fractal_video = None

    def generate_fractal(self,input_video='test.mp4',output_video='test_out.mp4',num_iterations=5, shift=0.1,scale=2,flip=False):
        # Load the video
        cap = cv2.VideoCapture(input_video)

        # Get video properties
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        aspect_ratio = width / height
        x = np.linspace(-scale * aspect_ratio, scale * aspect_ratio, width)
        y = np.linspace(-scale, scale, height)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y

        # Apply complex function
        for _ in range(num_iterations):
            # Shift the complex plane
            Z = Z - shift

            # Apply a transformation (e.g., square root)
            Z = Z ** 2
        #Z = self.normalise_complex(Z) 
        Z_transformed = Z 


        X_new = ((Z_transformed.real + scale * aspect_ratio) / (2 * scale * aspect_ratio)) * (width - 1)
        Y_new = ((Z_transformed.imag + scale) / (2 * scale)) * (height - 1)
        # Convert to float32 for OpenCV
        map_x = X_new.astype(np.float32)
        map_y = Y_new.astype(np.float32)
        map_x = np.clip(map_x, 0, width - 1).astype(np.float32)
        map_y = np.clip(map_y, 0, height - 1).astype(np.float32)
        
        for _ in tqdm(range(frame_count), desc="Processing video",unit='frames'):
            ret, frame = cap.read()
            if not ret:
                break
            
            if type(flip) is int:
                
                frame = cv2.flip(frame, flip)

            # Remap image
            frame = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

            # Show result
    

    
            self.fractal_exist=True

            # Write modified frame to output
            out.write(frame)

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        

    def save_fractal(self, output_path):
        if self.fractal_video:
            with open(output_path, 'w') as file:
                file.write(self.fractal_video)
            print(f"Fractal saved to {output_path}")
        else:
            print("No fractal generated yet.")
class audio_fractal:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.fractal_audio = None

    def generate_fractal(self):
        # Placeholder for fractal generation logic
        self.fractal_audio = f"Fractal generated from {self.audio_path}"

    def save_fractal(self, output_path):
        if self.fractal_audio:
            with open(output_path, 'w') as file:
                file.write(self.fractal_audio)
            print(f"Fractal saved to {output_path}")
        else:
            print("No fractal generated yet.")
class text_fractal:
    def __init__(self, text_path):
        self.text_path = text_path
        self.fractal_text = None

    def generate_fractal(self):
        # Placeholder for fractal generation logic
        self.fractal_text = f"Fractal generated from {self.text_path}"

    def save_fractal(self, output_path):
        if self.fractal_text:
            with open(output_path, 'w') as file:
                file.write(self.fractal_text)
            print(f"Fractal saved to {output_path}")
        else:
            print("No fractal generated yet.")
class generic_fractal:
    def __init__(self, file_path):
        self.file_path = file_path
        self.fractal_data = None

    def generate_fractal(self):
        # Placeholder for fractal generation logic
        self.fractal_data = f"Fractal generated from {self.file_path}"

    def save_fractal(self, output_path):
        if self.fractal_data:
            with open(output_path, 'w') as file:
                file.write(self.fractal_data)
            print(f"Fractal saved to {output_path}")
        else:
            print("No fractal generated yet.")