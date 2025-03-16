import unittest
import os
import tempfile
from PIL import Image
from coc.util.vision import generate_video

class TestGenerateVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Check if sample images exist in 'data/sample/'
        base_dir = os.path.join("data", "sample")
        required_files = [os.path.join(base_dir, f"img{i}.png") for i in range(4)]
        missing_files = [f for f in required_files if not os.path.isfile(f)]

        if missing_files:
            raise unittest.SkipTest(f"Missing sample images: {missing_files}")

        # Load images into PIL.Image objects
        cls.sample_images = [Image.open(f) for f in required_files]

    @classmethod
    def tearDownClass(cls):
        # Close all opened images
        for img in cls.sample_images:
            img.close()

    def test_generate_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = os.getcwd()
            try:
                os.chdir(tmpdir)
                output_filename = "test_output.mp4"

                # Generate video
                generate_video(self.sample_images, output_filename)

                # Verify output file exists and is non-empty
                self.assertTrue(os.path.isfile(output_filename))
                self.assertGreater(os.path.getsize(output_filename), 0)

                # Ensure temporary image files were cleaned up
                temp_files = [f for f in os.listdir() if f.startswith("temp_img")]
                self.assertEqual(len(temp_files), 0, "Temporary files were not cleaned up")

            finally:
                os.chdir(original_dir)

    def test_empty_images_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = os.getcwd()
            try:
                os.chdir(tmpdir)
                output_filename = "should_not_exist.mp4"

                # Expect error when images list is empty
                with self.assertRaises(ValueError):
                    generate_video([], output_filename)

                # Ensure no files were created
                self.assertFalse(os.path.exists(output_filename))
                temp_files = [f for f in os.listdir() if f.startswith("temp_img")]
                self.assertEqual(len(temp_files), 0)

            finally:
                os.chdir(original_dir)

if __name__ == "__main__":
    unittest.main(failfast=True)