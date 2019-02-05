import cv2
from src.utilities.image.image import sample_to_batch

batch_size = 5
elephant = cv2.imread("../TestImages/frames/frame_0002.png", cv2.IMREAD_COLOR)

random_batch = sample_to_batch(
    elephant,
    target_shape=[100, 100],
    upscale_to_target=False,
    batch_size=batch_size,
    always_sample_center=False)

for i in range(batch_size):
    cv2.imshow("mini", random_batch[i])
    cv2.waitKey(0)

random_batch_max = sample_to_batch(
    elephant,
    use_min_dimension=True,
    always_sample_center=True,
    batch_size=batch_size)

for i in range(batch_size):
    cv2.imshow("mini", random_batch_max[i])
    cv2.waitKey(0)