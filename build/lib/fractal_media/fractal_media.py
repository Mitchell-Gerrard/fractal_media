import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
class photo_fractal:
    def __init__(self, image_path):
        self.image_path = image_path
        self.fractal_image = None
    def normalise_complex(self,Z):
        # Normalize real and imaginary parts independently to [-1,1]
        real = Z.real
        imag = Z.imag
        
        real_norm = 2 * (real - real.min()) / (real.max() - real.min()) - 1
        imag_norm = 2 * (imag - imag.min()) / (imag.max() - imag.min()) - 1
        
        return real_norm + 1j * imag_norm
    def generate_squared_fractal(self,num_iterations=5,shift=0.1):
        # Load image
        img = np.array(Image.open(self.image_path).convert("RGB"))

        height, width = img.shape[:2]

        # Create coordinate grid (centered)
        aspect_ratio = width / height
        x = np.linspace(-aspect_ratio, aspect_ratio, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y

        # Apply complex function
        for _ in range(num_iterations):
            # Shift the complex plane
            Z = Z - shift

            # Apply a transformation (e.g., squaring)
            Z = np.sqrt(Z)
        #Z = self.normalise_complex(Z) 
        Z_transformed = Z 

        
        X_new = ((Z_transformed.real + 1) / 2) * (width - 1)
        Y_new = ((Z_transformed.imag + 1) / 2) * (height - 1)
        # Convert to float32 for OpenCV
        map_x = X_new.astype(np.float32)
        map_y = Y_new.astype(np.float32)

        # Remap image
        remapped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

        # Show result
        plt.imshow(remapped)
        plt.axis('off')
        plt.show()
        self.fractal_image = f"Fractal generated from {self.image_path}"
    def generate_square_root_fractal(self, num_iterations=5, shift=0.1):
        # Load image
        img = np.array(Image.open(self.image_path).convert("RGB"))

        height, width = img.shape[:2]

        aspect_ratio = width / height
        x = np.linspace(-2 * aspect_ratio, 2 * aspect_ratio, width)
        y = np.linspace(-2, 2, height)
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


        X_new = ((Z_transformed.real + 1) / 2) * (width - 1)
        Y_new = ((Z_transformed.imag + 1) / 2) * (height - 1)
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
        self.fractal_image = f"Fractal generated from {self.image_path}"
    def generate_constant_fractal(self, constant, num_iterations=5):
        # Load image
        img = np.array(Image.open(self.image_path).convert("RGB"))

        height, width = img.shape[:2]

        # Create coordinate grid (centered)
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        Z = X + 1j * Y

        # Apply complex function with constant
        for _ in range(num_iterations):
            Z = Z + constant
            Z = self.normalise_complex(Z) 

        Z_transformed = Z 

        # Extract and normalize real/imag back to pixel space
        X_new = ((Z_transformed.real - Z_transformed.real.min()) /
                (Z_transformed.real.max() - Z_transformed.real.min())) * (width - 1)
        Y_new = ((Z_transformed.imag - Z_transformed.imag.min()) /
                (Z_transformed.imag.max() - Z_transformed.imag.min())) * (height - 1)

        # Convert to float32 for OpenCV
        map_x = X_new.astype(np.float32)
        map_y = Y_new.astype(np.float32)

        # Remap image
        remapped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Show result
        plt.imshow(remapped)
        plt.axis('off')
        plt.show()
        self.fractal_image = f"Fractal generated from {self.image_path}"
    def save_fractal(self, output_path):
        if self.fractal_image:
            with open(output_path, 'w') as file:
                file.write(self.fractal_image)
            print(f"Fractal saved to {output_path}")
        else:
            print("No fractal generated yet.")
class video_fractal:
    def __init__(self, video_path):
        self.video_path = video_path
        self.fractal_video = None

    def generate_fractal(self):
        # Placeholder for fractal generation logic
        self.fractal_video = f"Fractal generated from {self.video_path}"

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