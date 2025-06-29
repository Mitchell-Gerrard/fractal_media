import fractal_media
import fractal_media.fractal_media

def test_fractal_media():
    image_fractal = fractal_media.fractal_media.photo_fractal("tests/test_image.jpg")
    image_fractal.generate_square_root_fractal(num_iterations=2, shift=-0.70176-0.3842j)
    image_fractal.generate_square_root_fractal(num_iterations=4, shift=0.5)

if __name__=="__main__":
    test_fractal_media()
    print("Fractal media test completed successfully.")