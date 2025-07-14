import fractal_media
import fractal_media.fractal_media

def test_fractal_media():
    image_fractal = fractal_media.fractal_media.photo_fractal("tests/test_image.jpg")
    image_fractal.generate_fractal(num_iterations=2, shift=-0.70176-0.3842j,scale=1)
    image_fractal.generate_fractal(num_iterations=2, shift=-0.2-0.3j,scale=1)
    image_fractal.save_fractal('tests/test_fractal.jpg',dpi=300)
    video_fractal = fractal_media.fractal_media.video_fractal()
    video_fractal.generate_fractal(input_video='tests/test_video.mp4', output_video='tests/test_out.mp4', num_iterations=2, shift=-0.4-0.3842j, scale=1, flip=1)



if __name__=="__main__":
    test_fractal_media()
    print("Fractal media test completed successfully.")