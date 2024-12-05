import os

validation_banana_path = './data/validation/banana'
banana_images = [f for f in os.listdir(validation_banana_path) if os.path.isfile(os.path.join(validation_banana_path, f))]
print(f"Number of images in 'banana' validation class: {len(banana_images)}")
